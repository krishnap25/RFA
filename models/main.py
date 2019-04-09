"""Main file to launch experiments."""

import argparse
import copy
import importlib
import gc
import random
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import time
from datetime import timedelta

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS, TRAINING_KEYS
from baseline_constants import OptimLoggingKeys
from baseline_constants import CORRUPTION_OMNISCIENT_KEY, CORRUPTION_FLIP_KEY, CORRUPTION_P_X_KEY
from baseline_constants import AGGR_MEAN, AGGR_GEO_MED
from client import Client
from model import ServerModel
from server import Server
from utils.constants import DATASETS
from utils.model_utils import read_data

SUMMARY_METRICS_PATH = 'output.log'


def main():
    args = parse_args()
    global_start_time = start_time = time.time()

    model_path = '{}/{}.py'.format(args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '{}.{}'.format(args.dataset, args.model)

    print('############################## {} ##############################'.format(model_path))
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Create 2 different models: one for server and one shared by all clients
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)
    else:
        args.lr = model_params[0]
    if args.decay_lr_every is None:
        args.decay_lr_every = 100 if args.dataset == 'femnist' else 50
    tf.reset_default_graph()
    print(args)
    client_model = ClientModel(*model_params, seed=args.seed)
    server_model = ServerModel(ClientModel(*model_params, seed=args.seed - 1))

    # Create server
    server = Server(server_model)

    # Create clients
    clients, corrupted_client_ids = setup_clients(args.dataset, model=client_model, validation=args.validation,
                                                  corruption=args.corruption, fraction_corrupt=args.fraction_corrupt,
                                                  seed=args.seed)
    print('#Clients = %d ; setup time = %s' % (len(clients), timedelta(seconds=round(time.time() - start_time))))
    print('Using true labels for corrupted data as well')
    gc.collect()  # free discarded memory in setup_clients

    # Logging utilities
    all_ids, all_groups, all_num_train_samples, all_num_test_samples = server.get_clients_info(clients)
    summary = None

    def log_helper(iteration, comm_rounds=None):
        if comm_rounds is None:
            comm_rounds = iteration
        nonlocal summary
        start_time = time.time()
        if args.no_logging:
            stat_metrics = None
        else:
            stat_metrics = server.test_model(clients, train_and_test=True)

        summary_iter = print_metrics(iteration, comm_rounds, stat_metrics,
                                     all_num_train_samples, all_num_test_samples,
                                     time.time() - start_time)
        if iteration == 0:
            summary = pd.Series(summary_iter).to_frame().T
        else:
            summary = summary.append(summary_iter, ignore_index=True)
            summary.to_csv(args.output_summary_file, mode='w', header=True, index=False)
        return summary_iter

    # Test untrained model on all clients
    s = log_helper(0)

    # Initialize diagnostics
    initial_loss = s.get(OptimLoggingKeys.TRAIN_LOSS_KEY, None)
    num_no_progress = 0

    # Simulate training
    for i in range(num_rounds):
        print('--- Round {} of {}: Training {} Clients ---'.format(i + 1, num_rounds, clients_per_round))
        sys.stdout.flush()
        start_time = time.time()

        # Select clients to train this round
        server.select_clients(online(clients), num_clients=clients_per_round)

        # Logging selection
        num_corr, num_cl, corr_frac = get_corrupted_fraction(server.selected_clients, corrupted_client_ids)
        print('\t\t\tCorrupted {:d} out of {:d} clients. {:.3f} fraction'.format(
            num_corr, num_cl, corr_frac), flush=True)

        # Simulate server model training on selected clients' data
        avg_loss, losses = server.train_model(num_epochs=args.num_epochs,
                                              batch_size=args.batch_size,
                                              minibatch=args.minibatch,
                                              lr=args.lr)

        # Update server model
        total_num_comm_rounds, is_updated = server.update_model(aggregation=args.aggregation,
                                                                corruption=args.corruption,
                                                                corrupted_client_ids=corrupted_client_ids,
                                                                maxiter=args.weiszfeld_maxiter)

        # Quit if no progress made
        if is_updated:
            num_no_progress = 0
        else:
            num_no_progress += 1
        if num_no_progress > args.patience_iter:
            print('No progress made in {} iterations. Quitting.'.format(num_no_progress))
            sys.exit(1)

        # Logging
        norm = _norm(server_model.model)
        print('\t\t\tRound: {} AvgLoss: {:.3f} Norm: {:.2f} Time: {} Tot_time {}'.format(
            i + 1, avg_loss, norm,
            timedelta(seconds=round(time.time() - start_time)),
            timedelta(seconds=round(time.time() - global_start_time))
        ), flush=True)

        # Test model on all clients
        if (
                (i + 1) == 5
                or (i + 1) % eval_every == 0
                or (i + 1) == num_rounds
                or (args.corruption == CORRUPTION_OMNISCIENT_KEY and (i + 1) < 10)
        ):
            s = log_helper(i + 1, total_num_comm_rounds)
            if OptimLoggingKeys.TRAIN_LOSS_KEY in s:
                if initial_loss is not None and s[OptimLoggingKeys.TRAIN_LOSS_KEY] > 3 * initial_loss:
                    print('Train_objective > 3 * initial_train_objective. Exiting')
                    break
            if args.no_logging:
                # save model
                save_model(server_model, args.dataset, args.model,
                           '{}_iteration{}'.format(args.output_summary_file, i + 1))

        if (i + 1) % args.decay_lr_every == 0:
            args.lr /= args.lr_decay

    # Save logs and server model
    summary.to_csv(args.output_summary_file, mode='w', header=True, index=False)
    save_model(server_model, args.dataset, args.model, args.output_summary_file)

    print('Job complete. Total time taken:', timedelta(seconds=round(time.time() - global_start_time)))

    # Close models
    server_model.close()
    client_model.close()


def _norm(model):
    with model.graph.as_default():
        weights = [np.linalg.norm(model.sess.run(v)) for v in tf.trainable_variables()]
    return np.linalg.norm(weights)


def online(clients):
    """We assume all users are always online."""
    return clients


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        required=True)
    parser.add_argument('-model',
                        help='name of model;',
                        type=str,
                        required=True)
    parser.add_argument('--num-rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval-every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients-per-round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--seed',
                        help='random seed for reproducibility;',
                        type=int)

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                                        help='None for FedAvg, else fraction;',
                                        type=float,
                                        default=None)
    epoch_capability_group.add_argument('--num_epochs',
                                        help='number of epochs when clients train on data;',
                                        type=int,
                                        default=1)

    parser.add_argument('-lr',
                        help='learning rate for local optimizers;',
                        type=float,
                        default=-1,
                        required=False)
    parser.add_argument('--lr-decay',
                        help='decay in learning rate, with frequency of decay specified by `--decay-lr-every`',
                        type=float,
                        default=1)
    parser.add_argument('--decay-lr-every',
                        help='frequency of decay of learning rate specified in number of epochs',
                        type=int)
    parser.add_argument('--output_summary_file',
                        help='Filename to log summary of optimization performance in CSV',
                        default=SUMMARY_METRICS_PATH)
    parser.add_argument('--validation',
                        help='If specified, hold out part of training data to use as a dev set for parameter search',
                        action='store_true')
    parser.add_argument('--patience-iter',
                        help='Number of patience rounds of no updates to wait for before quitting',
                        type=int,
                        default=20)
    parser.add_argument('--corruption',
                        help=""""Corrupt data in any clients? If not specified, add no corruptions.
                        Choices '{}' or '{}' manipulate the data of the corrupt devices, while choice '{}'
                        leads to corrupt devices to propose an update leading to negation of the update.
                        """.format(CORRUPTION_FLIP_KEY, CORRUPTION_P_X_KEY, CORRUPTION_OMNISCIENT_KEY),
                        choices=[CORRUPTION_OMNISCIENT_KEY, CORRUPTION_FLIP_KEY, CORRUPTION_P_X_KEY],
                        type=str)
    parser.add_argument('--fraction-corrupt',
                        help="""Fraction of data to corrupt.
                        Chooses clients randomly until total fraction of corrupt data has just exceeded
                        specified fraction""",
                        type=float,
                        default=0.1)
    parser.add_argument('--aggregation',
                        help='Aggregation technique used to combine updates or gradients',
                        choices=[AGGR_MEAN, AGGR_GEO_MED],
                        default=AGGR_MEAN)
    parser.add_argument('--weiszfeld-maxiter',
                        help="""Number of Weiszfeld iterations used to compute when `--aggregation` 
                        is chosen as {}""".format(AGGR_GEO_MED),
                        type=int,
                        default=4)
    parser.add_argument('--no-logging',
                        help='if specified, do not perform testing. Instead save model to disk.',
                        action='store_true')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, sys.maxsize)
        print('Random seed not provided. Using {} as seed'.format(args.seed))

    return args


def setup_clients(dataset, model=None, validation=False, corruption=None,
                  fraction_corrupt=0.1, seed=-1, subsample=True):
    """Instantiates clients based on given train and test data directories.
        If validation is True, use part of training set as validation set

    Return:
        all_clients: list of Client objects.
    """
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', 'test')

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    # subsample
    if subsample and dataset == 'femnist':
        # Pick 1000 fixed users for experiment
        rng_sub = random.Random(25)
        users = rng_sub.sample(users, 1000)
        train_data = {u: p for (u, p) in train_data.items() if u in users}
        test_data = {u: p for (u, p) in test_data.items() if u in users}
        # note: groups are empty for femnist

    if validation:  # split training set into train and val in the ratio 80:20
        print('Validation mode, splitting train data into train and val sets...')
        for idx, u in enumerate(users):
            data = list(zip(train_data[u]['x'], train_data[u]['y']))
            rng = random.Random(idx)
            rng.shuffle(data)
            split_point = int(0.8 * len(data))
            x, y = zip(*data[:split_point])
            x1, y1 = zip(*data[split_point:])
            train_data[u] = {'x': list(x), 'y': list(y)}
            test_data[u] = {'x': list(x1), 'y': list(y1)}
    if len(groups) == 0:
        groups = [[] for _ in users]
    all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    corrupted_clients = apply_corruption_all(all_clients, dataset, corruption, fraction_corrupt, seed)
    return all_clients, corrupted_clients


def corrupt_one_client_data(dataset, client, corruption):
    """Apply corruption to train data of given client in-place. Note: eval data is unchanged"""
    x = client.train_data['x']
    y = client.train_data['y']
    if dataset == 'femnist':
        if corruption == CORRUPTION_FLIP_KEY:
            # flip labels, leave images untouched
            client.train_data['y_true'] = copy.deepcopy(y)
            for i in range(len(y)):
                if y[i] < 10:
                    # digit: apply deterministic permutation
                    y[i] = (7 * y[i] + 1) % 10
                elif y[i] < 36:
                    # upper case letter: convert to lower case
                    y[i] += 26
                else:
                    # lower case letter: convert to upper case
                    y[i] -= 26
        elif corruption == CORRUPTION_P_X_KEY:
            # take negative of image, leave labels untouched
            x_new = []
            for img in x:
                x_new.append([1 - pixel for pixel in img])
            client.train_data['x_true'] = x
            client.train_data['x'] = x_new

        else:
            raise ValueError('Unknown corruption, {}'.format(corruption))
    elif dataset == 'sent140':
        for i in range(len(y)):
            y[i] = 1 - y[i]  # flip 0-1 labels
    elif dataset == 'shakespeare':
        x_new = []
        y_new = []
        for i in range(len(y)):
            # reverse sentence
            s = x[i] + y[i]
            s = s[::-1]
            x_new.append(s[:-1])
            y_new.append(s[-1])
        # modify client in-place
        client.train_data['x_true'] = x
        client.train_data['y_true'] = y
        client.train_data['x'] = x_new
        client.train_data['y'] = y_new


def apply_corruption_all(client_list, dataset, corruption, fraction_corrupt, seed):
    """Apply corruptions to clients so that a total of `fraction_corrupt` fraction of the data is corrupted.
        Return list of corrupt clients and modify client_list inplace.
    """
    if corruption:
        client_dict = {client.id: client for client in client_list}
        rng = random.Random(seed - 1)
        users = [client.id for client in client_list]
        rng.shuffle(users)

        # choose prefix of `users` to corrupt until fraction has just been exceeded
        num_data_pts = [len(client_dict[u].train_data['y']) for u in users]
        total_num_data_pts = sum(num_data_pts)
        target_num_data_pts = fraction_corrupt * total_num_data_pts  # number of data points to corrupt
        num_corrupted_data_pts = 0
        end_idx = 0  # exclusive
        while num_corrupted_data_pts < target_num_data_pts:
            num_corrupted_data_pts += num_data_pts[end_idx]
            end_idx += 1
        corrupted_clients = users[:end_idx]
        print('Corrupting {:0.4f} fraction of data'.format(num_corrupted_data_pts / total_num_data_pts))

        # flip labels if need be
        if corruption in [CORRUPTION_FLIP_KEY, CORRUPTION_P_X_KEY]:
            for u in corrupted_clients:
                corrupt_one_client_data(dataset, client_dict[u], corruption)

    else:
        corrupted_clients = []

    return frozenset(corrupted_clients)


def get_corrupted_fraction(selected_clients, corrupted_client_ids):
    total_num_pts = sum([len(c.train_data['y']) for c in selected_clients])
    corrupted_lens = [len(c.train_data['y']) for c in selected_clients
                      if c.id in corrupted_client_ids]
    num_corrupted_clients = len(corrupted_lens)
    num_corrupted_pts = sum(corrupted_lens)
    return (num_corrupted_clients, len(selected_clients),
            num_corrupted_pts / total_num_pts)


def save_model(server_model, dataset, model, output_summary_file):
    """Saves the given server model on checkpoints/dataset/model.ckpt."""
    # Save server model
    start_time = time.time()
    ckpt_path = os.path.join('checkpoints', *(output_summary_file.split(os.path.sep)[1:]))
    print(ckpt_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server_model.save('%s.ckpt' % ckpt_path)
    print('Model saved in path: {} in time {:.2f} sec'.format(save_path, time.time() - start_time))


def print_metrics(iteration, comm_rounds, metrics, train_weights, test_weights, elapsed_time=0):
    """Prints weighted averages of the given metrics.

    Args:
        iteration: current iteration number
        comm_rounds: number of communication rounds
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        train_weights: dict with client ids as keys. Each entry is the weight
            for that client for training metrics.
        test_weights: dict with client ids as keys. Each entry is the weight
            for that client for testing metrics
        elapsed_time: time taken for testing
    """
    output = {'iteration': iteration, 'comm_rounds': comm_rounds}
    if metrics is None:
        print(iteration, comm_rounds)
    else:
        print(iteration, end=', ')
        ordered_tr_weights = [train_weights[c] for c in sorted(train_weights)]
        ordered_te_weights = [test_weights[c] for c in sorted(test_weights)]
        metric_names = get_metrics_names(metrics)
        for metric in metric_names:
            ordered_weights = ordered_tr_weights if metric in TRAINING_KEYS else ordered_te_weights
            ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
            avg_metric = np.average(ordered_metric, weights=ordered_weights)
            output[metric] = avg_metric
            print('%s: %g' % (metric, avg_metric), end=', ')
        print('Time:', timedelta(seconds=round(elapsed_time)))
    sys.stdout.flush()
    return output


def get_metrics_names(metrics):
    """Gets the names of the metrics.

    Args:
        metrics: Dict keyed by client id. Each element is a dict of metrics
            for that client in the specified round. The dicts for all clients
            are expected to have the same set of keys."""
    if len(metrics) == 0:
        return []
    metrics_dict = next(iter(metrics.values()))
    return list(metrics_dict.keys())


if __name__ == '__main__':
    main()
