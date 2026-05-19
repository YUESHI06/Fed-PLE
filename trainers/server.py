import torch
import copy
import numpy as np


class Server:
    """Base federated server with FedAvg aggregation."""
    def __init__(self, args, global_model, device, criterion):
        self.args = args
        self.global_model = global_model
        self.device = device
        self.criterion = criterion
        self.result_dict = {}

    def initialize_epoch_updates(self, epoch):
        self.epoch = epoch
        self.model_updates = []
        self.num_samples_list = []
        self.result_dict[self.epoch] = {'train': [], 'test': []}

    def save_train_updates(self, model_state, num_sample, result):
        self.model_updates.append(model_state)
        self.num_samples_list.append(num_sample)
        self.result_dict[self.epoch]['train'].append(result)

    def average_weights(self):
        """FedAvg aggregation weighted by the number of local samples."""
        if len(self.model_updates) == 0:
            return
        sample_sum = float(sum(self.num_samples_list)) if self.num_samples_list else 0.0
        if sample_sum <= 0:
            weights = [1.0 / len(self.model_updates)] * len(self.model_updates)
        else:
            weights = [n / sample_sum for n in self.num_samples_list]
        w_avg = copy.deepcopy(self.model_updates[0])
        for key in w_avg.keys():
            w_avg[key] = self.model_updates[0][key] * weights[0]
        for key in w_avg.keys():
            for i in range(1, len(self.model_updates)):
                w_avg[key] += self.model_updates[i][key] * weights[i]
        self.global_model.load_state_dict(copy.deepcopy(w_avg))


class ARFL_Server(Server):
    """Adaptive Robust Federated Learning server."""
    def __init__(self, args, global_model, criterion, clients, total_num_samples):
        super().__init__(args, global_model, args.device, criterion)
        self.clients = clients
        self.client_num = args.client_num
        self.total_num_samples = total_num_samples
        self.reg_weight = (
            self.total_num_samples if args.reg_weight is None
            else args.reg_weight * self.total_num_samples
        )

    def sample_clients(self, my_round):
        candidates = list(range(self.client_num))
        while True:
            selected = np.random.choice(
                candidates, int(self.client_num * self.args.sample_rate), replace=False
            ).tolist()
            if sum(self.clients[c].weight for c in selected) != 0:
                break
        self.selected_clients = [self.clients[idx] for idx in selected]
        print(f"ARFL Round {my_round}: selected clients {selected}")

    def average_weights(self):
        weights = [c.weight for c in self.selected_clients]
        if sum(weights) > 0:
            nor_weights = np.array(weights) / np.sum(weights)
            first_params = self.selected_clients[0].get_model_parameters()
            w_avg = copy.deepcopy(first_params)
            for key in w_avg.keys():
                w_avg[key] = first_params[key] * nor_weights[0]
            for key in w_avg.keys():
                for i in range(1, len(self.selected_clients)):
                    params = self.selected_clients[i].get_model_parameters()
                    w_avg[key] += params[key] * nor_weights[i]
            self.global_model.load_state_dict(copy.deepcopy(w_avg))

    def update_alpha(self):
        for c in self.selected_clients:
            c.test()
        idxs = [x for x, _ in sorted(enumerate(self.clients),
                                       key=lambda x: x[1].get_test_loss())]
        eta_optimal = self.clients[idxs[0]].get_test_loss() + self.reg_weight
        for p in range(len(idxs)):
            eta = (
                sum(self.clients[i].num_train_samples * self.clients[i].get_test_loss()
                    for i in idxs[:p+1]) + self.reg_weight
            ) / sum(self.clients[i].num_train_samples for i in idxs[:p+1])
            if eta - self.clients[idxs[p]].get_test_loss() < 0:
                break
            eta_optimal = eta
        for c in self.clients:
            w = c.num_train_samples * max(eta_optimal - c.get_test_loss(), 0) / self.reg_weight
            c.set_weight(w)


class CLC_Server(Server):
    """Collaborative Label Correction server."""
    def __init__(self, args, global_model, device, criterion):
        super().__init__(args, global_model, device, criterion)
        self.class_nums_each = [[] for _ in range(args.client_num)]
        self.conflist_each = [[] for _ in range(args.client_num)]

    def receiveconf(self, confs, classnums):
        for ix in range(self.args.client_num):
            self.conflist_each[ix] = confs[ix]
            self.class_nums_each[ix] = classnums[ix]

    def conf_agg(self):
        conf_score = [0] * self.args.num_classes
        class_nums = np.array(self.class_nums_each)
        sum_col = class_nums.sum(axis=0)
        for ix in range(self.args.client_num):
            for i in range(self.args.num_classes):
                denom = sum_col[i] if sum_col[i] > 0 else 1
                w = self.class_nums_each[ix][i] / denom
                conf_score[i] += w * self.conflist_each[ix][i]
        return conf_score
