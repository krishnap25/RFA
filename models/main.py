"""Main file to launch experiments."""

import importlib
import gc
import os
import sys

import pandas as pd
import tensorflow as tf
import time
from datetime import timedelta

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from baseline_constants import OptimLoggingKeys
from model import ServerModel
from server import Server
from utils import model_utils as utils


def main():
    args = utils.parse_args()
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
    clients, corrupted_client_ids = utils.setup_clients(args.dataset, args.data_dir, model=client_model,
                                                        validation=args.validation, seed=args.seed,
                                                        corruption=args.corruption,
                                                        fraction_corrupt=args.fraction_corrupt)
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

        summary_iter = utils.print_metrics(iteration, comm_rounds, stat_metrics,
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
        server.select_clients(utils.online(clients), num_clients=clients_per_round)

        # Logging selection
        num_corr, num_cl, corr_frac = utils.get_corrupted_fraction(server.selected_clients, corrupted_client_ids)
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
        norm = utils.model_norm(server_model.model)
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
                utils.save_model(server_model, args.dataset, args.model,
                                 '{}_iteration{}'.format(args.output_summary_file, i + 1))

        if (i + 1) % args.decay_lr_every == 0:
            args.lr /= args.lr_decay

    # Save logs and server model
    summary.to_csv(args.output_summary_file, mode='w', header=True, index=False)
    utils.save_model(server_model, args.dataset, args.model, args.output_summary_file)

    print('Job complete. Total time taken:', timedelta(seconds=round(time.time() - global_start_time)))

    # Close models
    server_model.close()
    client_model.close()


if __name__ == '__main__':
    main()
