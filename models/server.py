import numpy as np

from baseline_constants import CORRUPTION_OMNISCIENT_KEY, MAX_UPDATE_NORM


class Server:

    def __init__(self, model):
        self.model = model  # global model of the server.
        self.selected_clients = []
        self.updates = []
        self.rng = model.rng  # use random number generator of the model
        self.total_num_comm_rounds = 0

    def select_clients(self, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = self.rng.sample(possible_clients, num_clients)

        return [(len(c.train_data['y']), len(c.eval_data['y'])) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, lr=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            lr: learning rate to use
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        losses = []
        for c in clients:
            self.model.send_to([c])  # reset client model

            comp, num_samples, averaged_loss, update = c.train(num_epochs, batch_size, minibatch, lr)
            losses.append(averaged_loss)

            self.updates.append((num_samples, update))

        return np.average(losses, weights=[len(c.train_data['y']) for c in clients]), losses

    def update_model(self, aggregation, corruption=None, corrupted_client_ids=frozenset(), maxiter=4):
        is_corrupted = [(client.id in corrupted_client_ids) for client in self.selected_clients]
        if corruption == CORRUPTION_OMNISCIENT_KEY and any(is_corrupted):
            # compute omniscient update
            avg = self.model.weighted_average_oracle([u[1] for u in self.updates], [u[0] for u in self.updates])
            num_pts = sum([u[0] for u in self.updates])

            corrupted_updates = [u for c, u in zip(is_corrupted, self.updates) if c]
            corrupted_avg = self.model.weighted_average_oracle([u[1] for u in corrupted_updates],
                                                               [u[0] for u in corrupted_updates])
            num_corrupt_pts = sum([u[0] for u in corrupted_updates])
            omniscient_update = [wc - 2 * num_pts / num_corrupt_pts * w_avg for wc, w_avg in zip(corrupted_avg, avg)]
            # change self.updates to reflect omniscient update
            for i in range(len(self.updates)):
                if is_corrupted[i]:
                    self.updates[i] = (self.updates[i][0], omniscient_update)

        num_comm_rounds, is_updated = self.model.update(self.updates, aggregation,
                                                        max_update_norm=MAX_UPDATE_NORM,
                                                        maxiter=maxiter)
        self.total_num_comm_rounds += num_comm_rounds
        self.updates = []
        return self.total_num_comm_rounds, is_updated

    def test_model(self, clients_to_test=None, train_and_test=False):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            train_and_test: If True, also measure metrics on training data
        """
        if clients_to_test is None:
            clients_to_test = self.selected_clients
        metrics = {}

        self.model.send_to(clients_to_test)

        for client in clients_to_test:
            c_metrics = client.test(self.model.cur_model, train_and_test)
            metrics[client.id] = c_metrics

        return metrics

    def get_clients_info(self, clients=None):
        """Returns the ids, hierarchies, num_train_samples and num_test_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients
        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_train_samples = {c.id: c.num_train_samples for c in clients}
        num_test_samples = {c.id: c.num_test_samples for c in clients}
        return ids, groups, num_train_samples, num_test_samples
