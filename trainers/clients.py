"""
Federated client trainers for all methods using CEGT model.
Each client operates on graph data: (node_features, adj, labels)
"""

import gc
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_processing.graph_dataset import GraphDataset, GraphNoiseDataset, collate_graph_batch


# ============================================================
# FedAvg Client (also base for CL, FedCorr)
# ============================================================
class FedAvg_Client:
    def __init__(self, args, criterion, model, dataloader):
        self.args = args
        self.criterion = criterion
        self.model = model
        self.dataloader = dataloader
        self.device = args.device

    def get_parameters(self):
        return self.model.state_dict()

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.inner_lr)
        self.result = {'sample': len(self.dataloader.dataset), 'loss': 0}
        self.model.train()
        for epoch in range(self.args.local_epoch):
            for x, adj, labels in self.dataloader:
                optimizer.zero_grad()
                x, adj, labels = x.to(self.device), adj.to(self.device), labels.to(self.device)
                outputs = self.model(x, adj)
                labels = labels.long().flatten()
                loss = self.criterion(outputs, labels)
                self.result['loss'] += loss.item()
                loss.backward()
                optimizer.step()
                del x, adj, labels, outputs, loss
                torch.cuda.empty_cache()


# ============================================================
# ARFL Client
# ============================================================
class ARFL_Client:
    def __init__(self, args, criterion, model, dataloader, weight, num_train_samples):
        self.args = args
        self.device = args.device
        self.criterion = criterion
        self.model = model
        self.dataloader = dataloader
        self.weight = weight
        self.num_train_samples = num_train_samples
        self.test_loss = 0

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.inner_lr)
        self.result = {'sample': len(self.dataloader.dataset), 'loss': 0}
        for epoch in range(self.args.local_epoch):
            self.model.train()
            for x, adj, labels in self.dataloader:
                optimizer.zero_grad()
                x, adj, labels = x.to(self.device), adj.to(self.device), labels.to(self.device)
                outputs = self.model(x, adj)
                labels = labels.long().flatten()
                loss = self.criterion(outputs, labels)
                self.result['loss'] += loss.item()
                loss.backward()
                optimizer.step()
                del x, adj, labels, outputs, loss
                torch.cuda.empty_cache()

    def test(self):
        self.test_loss = 0
        self.model.eval()
        with torch.no_grad():
            for x, adj, labels in self.dataloader:
                x, adj, labels = x.to(self.device), adj.to(self.device), labels.to(self.device)
                outputs = self.model(x, adj)
                labels = labels.long().flatten()
                loss = self.criterion(outputs, labels)
                self.test_loss += loss.item()
                del x, adj, labels, outputs, loss
                torch.cuda.empty_cache()

    def get_model_parameters(self):
        return self.model.state_dict()

    def get_test_loss(self):
        return self.test_loss

    def set_weight(self, weight):
        self.weight = weight


# ============================================================
# CL (Confidence Learning) Client
# ============================================================
class CL_Client(FedAvg_Client):
    """Uses cross-validation to filter noisy samples."""
    def __init__(self, args, criterion, model, dataloader):
        super().__init__(args, criterion, model, dataloader)

    def cross_validation(self, dl_1, dl_2):
        """Train two models on two halves, keep agreed-upon samples."""
        model_1 = copy.deepcopy(self.model)
        model_2 = copy.deepcopy(self.model)
        opt_1 = torch.optim.Adam(model_1.parameters(), lr=self.args.inner_lr)
        opt_2 = torch.optim.Adam(model_2.parameters(), lr=self.args.inner_lr)

        # Train model_1 on dl_1
        model_1.train()
        for e in range(20):
            for x, adj, y in dl_1:
                opt_1.zero_grad()
                x, adj, y = x.to(self.device), adj.to(self.device), y.to(self.device)
                pred = model_1(x, adj)
                loss = self.criterion(pred, y.long().flatten())
                loss.backward()
                opt_1.step()

        # Train model_2 on dl_2
        model_2.train()
        for e in range(20):
            for x, adj, y in dl_2:
                opt_2.zero_grad()
                x, adj, y = x.to(self.device), adj.to(self.device), y.to(self.device)
                pred = model_2(x, adj)
                loss = self.criterion(pred, y.long().flatten())
                loss.backward()
                opt_2.step()

        # Cross-validate: keep samples both models agree on
        model_1.eval()
        model_2.eval()
        clean_data = []
        with torch.no_grad():
            for x, adj, y in dl_2:
                x, adj, y = x.to(self.device), adj.to(self.device), y.to(self.device)
                pred = F.softmax(model_1(x, adj), dim=-1)
                preds = torch.argmax(pred, dim=-1).long()
                y_flat = y.long().flatten()
                indices = torch.nonzero(torch.eq(preds, y_flat), as_tuple=False).squeeze(dim=1)
                if indices.numel() > 0:
                    for idx in indices:
                        clean_data.append((x[idx].cpu(), adj[idx].cpu(), y_flat[idx].cpu()))

            for x, adj, y in dl_1:
                x, adj, y = x.to(self.device), adj.to(self.device), y.to(self.device)
                pred = F.softmax(model_2(x, adj), dim=-1)
                preds = torch.argmax(pred, dim=-1).long()
                y_flat = y.long().flatten()
                indices = torch.nonzero(torch.eq(preds, y_flat), as_tuple=False).squeeze(dim=1)
                if indices.numel() > 0:
                    for idx in indices:
                        clean_data.append((x[idx].cpu(), adj[idx].cpu(), y_flat[idx].cpu()))

        if clean_data:
            # Rebuild dataloader from clean data
            clean_ds = _ListDataset(clean_data)
            self.dataloader = DataLoader(clean_ds, batch_size=self.args.batch,
                                         shuffle=True, collate_fn=collate_graph_batch)
        return self.dataloader


# ============================================================
# FedCorr Client
# ============================================================
class FedCorr_Client(FedAvg_Client):
    """FedCorr: multi-stage training with noise detection."""
    def __init__(self, args, criterion, model, dataloader):
        super().__init__(args, criterion, model, dataloader)

    def get_output(self):
        """Get model outputs and losses for noise estimation."""
        self.model.eval()
        outputs_whole = None
        loss_whole = None
        with torch.no_grad():
            for x, adj, y in self.dataloader:
                x, adj, y = x.to(self.device), adj.to(self.device), y.to(self.device)
                outputs = self.model(x, adj)
                outputs = F.softmax(outputs, dim=1)
                y = y.long().flatten()
                loss = self.criterion(outputs, y)
                if outputs_whole is None:
                    outputs_whole = outputs.cpu().numpy()
                    loss_whole = np.array([loss.cpu().item()])
                else:
                    outputs_whole = np.concatenate((outputs_whole, outputs.cpu().numpy()), axis=0)
                    loss_whole = np.concatenate((loss_whole, np.array([loss.cpu().item()])))
        return outputs_whole, loss_whole


# ============================================================
# CLC (Collaborative Label Correction) Client
# ============================================================
class CLC_Client:
    def __init__(self, args, criterion, model, dataset, client_id, tao):
        self.args = args
        self.criterion = criterion
        self.model = model
        self.dataset = dataset
        self.client_id = client_id
        self.tao = tao
        self.device = args.device
        self.dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                                      collate_fn=collate_graph_batch)

    def get_parameters(self):
        return self.model.state_dict()

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.inner_lr)
        self.result = {'sample': len(self.dataloader.dataset), 'loss': 0}
        self.model.train()
        for epoch in range(self.args.local_epoch):
            for x, adj, y in self.dataloader:
                optimizer.zero_grad()
                x, adj, y = x.to(self.device), adj.to(self.device), y.to(self.device)
                outputs = self.model(x, adj)
                y = y.long().flatten()
                loss = self.criterion(outputs, y)
                self.result['loss'] += loss.item()
                loss.backward()
                optimizer.step()
                del x, adj, y, outputs, loss
                torch.cuda.empty_cache()

    def sendconf(self):
        confListU, class_nums = self.confidence()
        return confListU, class_nums

    def confidence(self):
        outputSofma = self._outputSof()
        r = outputSofma.shape[0]
        c = outputSofma.shape[1]
        prob_everyclass = [[] for _ in range(c - 1)]
        class_nums = []
        confList = []
        for i in range(r):
            oriL = int(outputSofma[i][c - 1])
            pro = outputSofma[i, oriL]
            prob_everyclass[oriL].append(pro)
        for i in range(c - 1):
            if len(prob_everyclass[i]) > 0:
                confList.append(round(np.mean(prob_everyclass[i], axis=0), 3))
            else:
                confList.append(0.0)
            class_nums.append(len(prob_everyclass[i]))
        self.sfm_Mat = outputSofma
        return confList, class_nums

    def _outputSof(self):
        self.model.eval()
        outputs_list = []
        labels_list = []
        eval_loader = DataLoader(self.dataset, batch_size=self.args.batch, shuffle=False,
                                 collate_fn=collate_graph_batch)
        with torch.no_grad():
            for x, adj, y in eval_loader:
                x, adj = x.to(self.device), adj.to(self.device)
                out = self.model(x, adj)
                outputs_list.append(out.cpu())
                labels_list.append(y.long().flatten().cpu())
        all_outputs = torch.cat(outputs_list, dim=0)
        labels_np = torch.cat(labels_list, dim=0).numpy()
        psx = F.softmax(all_outputs, dim=1).numpy()
        s = labels_np.reshape(-1, 1)
        sfm_Mat = np.hstack((psx, s))
        return sfm_Mat

    def data_holdout(self, conf_score):
        r = self.sfm_Mat.shape[0]
        delta_sort = {}
        self.keys = []
        self.sudo_labels = []
        for idx in range(r):
            softmax = self.sfm_Mat[idx]
            maxPro_Naive, preIndex_Naive = -1, -1
            maxPro, preIndex = -1, -1
            for j in range(self.args.num_classes):
                if softmax[j] > maxPro_Naive:
                    preIndex_Naive = j
                    maxPro_Naive = softmax[j]
                if softmax[j] > conf_score[j]:
                    if softmax[j] > maxPro:
                        maxPro = softmax[j]
                        preIndex = j
            label = int(softmax[-1])
            margin = maxPro_Naive - softmax[label]
            if preIndex == -1:
                preIndex = preIndex_Naive
            elif preIndex != label:
                delta_sort[idx] = margin
            self.sudo_labels.append(preIndex)

        delta_sorted = sorted(delta_sort.items(), key=lambda x: x[1], reverse=True)
        for (k, v) in delta_sorted:
            if v > self.tao:
                self.keys.append(k)

        reserve = [idx for idx in range(r) if idx not in self.keys]
        self.sudo_labels = [self.sudo_labels[i] for i in reserve]
        self.dataset.labels = list(self.sudo_labels)
        self.dataset.node_features = [self.dataset.node_features[i] for i in reserve]
        self.dataset.adj_matrices = [self.dataset.adj_matrices[i] for i in reserve]
        self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch,
                                      shuffle=True, collate_fn=collate_graph_batch)

    def data_correct(self):
        if len(self.sudo_labels) == len(self.dataset.labels):
            self.dataset.labels = list(self.sudo_labels)
        self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch,
                                      shuffle=True, collate_fn=collate_graph_batch)


# ============================================================
# RESCUER (PLE) Client - Meta-learning with LCN
# ============================================================
class RESCUER_Client:
    """
    RESCUER client using CEGT (inner model) + LCN (outer model).
    Implements probabilistic label estimation via meta-learning.
    """
    def __init__(self, args, criterion, device, inner_model, outer_model,
                 noise_dataloader, pure_dataloader):
        self.args = args
        self.criterion = criterion
        self.device = device
        self.inner_model = inner_model
        self.outer_model = outer_model
        self.noise_dataloader = noise_dataloader
        self.pure_dataloader = pure_dataloader

    def get_inner_parameters(self):
        return self.inner_model.state_dict()

    def print_loss(self):
        print(f"  outer_loss = {self.result.get('outer_loss', 'N/A')}")

    def meta_train(self):
        torch.autograd.set_detect_anomaly(True)
        inner_model_copy = copy.deepcopy(self.inner_model)
        outer_optimizer = torch.optim.Adam(self.outer_model.parameters(), lr=self.args.outer_lr)
        inner_optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.args.inner_lr)
        inner_copy_opt = torch.optim.Adam(inner_model_copy.parameters(), lr=self.args.inner_lr)

        self.result = {'sample': len(self.noise_dataloader.dataset), 'outer_loss': 0.}

        for epoch in range(self.args.local_epoch):
            # Phase 1: Train outer model (LCN) for 3 iterations
            for e in range(3):
                self.result['outer_loss'] = 0.
                for noise_batch, pure_batch in zip(self.noise_dataloader, self.pure_dataloader):
                    # Unpack noise batch (x, adj, noise_labels, global_labels)
                    x, adj, noise_labels, global_labels = (
                        noise_batch[0].to(self.device), noise_batch[1].to(self.device),
                        noise_batch[2].to(self.device), noise_batch[3].to(self.device)
                    )
                    # Unpack pure batch
                    px, padj, py = (
                        pure_batch[0].to(self.device), pure_batch[1].to(self.device),
                        pure_batch[2].to(self.device)
                    )

                    # Inner step: train copy model with LCN-corrected labels
                    inner_model_copy.train()
                    self.outer_model.eval()
                    inner_copy_opt.zero_grad()
                    predictions = inner_model_copy(x, adj)
                    predictions = F.log_softmax(predictions, dim=-1)

                    # Get intermediate features
                    h_x = inner_model_copy.inter_outputs
                    h_x = h_x.clone().requires_grad_(True)

                    # Build label pairs for LCN
                    gl_onehot = F.one_hot(global_labels.long().flatten(), num_classes=2).unsqueeze(1)
                    nl_onehot = F.one_hot(noise_labels.long().flatten(), num_classes=2).unsqueeze(1)
                    cat_labels = torch.cat((gl_onehot, nl_onehot), dim=1).float()
                    cat_labels = cat_labels.clone().requires_grad_(True)

                    outer_out = self.outer_model(h_x, cat_labels)
                    outer_out = torch.squeeze(outer_out, dim=1)
                    outer_out = torch.softmax(outer_out, dim=-1)

                    inner_loss = F.kl_div(predictions, outer_out, reduction='batchmean')
                    inner_loss.backward()
                    inner_copy_opt.step()

                    # Outer step: evaluate copy model on pure data
                    outer_optimizer.zero_grad()
                    inner_model_copy.train()
                    self.outer_model.train()
                    updated_pred = inner_model_copy(px, padj)
                    pure_labels = py.long().flatten()
                    outer_loss = self.criterion(updated_pred, pure_labels)
                    self.result['outer_loss'] += outer_loss.item()
                    outer_loss.backward()
                    outer_optimizer.step()

                    del x, adj, noise_labels, global_labels, h_x
                    del outer_out, cat_labels, inner_loss, outer_loss
                    torch.cuda.empty_cache()
                    gc.collect()

            # Phase 2: Update inner model with LCN labels
            for e in range(1):
                for noise_batch in self.noise_dataloader:
                    x, adj, noise_labels, global_labels = (
                        noise_batch[0].to(self.device), noise_batch[1].to(self.device),
                        noise_batch[2].to(self.device), noise_batch[3].to(self.device)
                    )
                    self.inner_model.train()
                    self.outer_model.eval()
                    inner_optimizer.zero_grad()
                    predictions = self.inner_model(x, adj)
                    predictions = F.log_softmax(predictions, dim=-1)

                    h_x = self.inner_model.inter_outputs
                    h_x = h_x.clone().requires_grad_(True)
                    gl_onehot = F.one_hot(global_labels.long().flatten(), num_classes=2).unsqueeze(1)
                    nl_onehot = F.one_hot(noise_labels.long().flatten(), num_classes=2).unsqueeze(1)
                    cat_labels = torch.cat((gl_onehot, nl_onehot), dim=1).float()
                    cat_labels = cat_labels.clone().requires_grad_(True)

                    outer_out = self.outer_model(h_x, cat_labels)
                    outer_out = torch.squeeze(outer_out, dim=1)
                    outer_out = torch.softmax(outer_out, dim=-1)

                    inner_loss = F.kl_div(predictions, outer_out, reduction='batchmean')
                    inner_loss.backward()
                    inner_optimizer.step()

                    del x, adj, noise_labels, global_labels, h_x
                    del outer_out, cat_labels, inner_loss
                    torch.cuda.empty_cache()
                    gc.collect()

    def warm_up(self):
        """Warm-up phase: train inner model on pure data only."""
        inner_optimizer = torch.optim.Adam(self.inner_model.parameters(), lr=self.args.inner_lr)
        self.result = {'sample': len(self.pure_dataloader.dataset)}
        self.inner_model.train()
        for x, adj, y in self.pure_dataloader:
            x, adj, y = x.to(self.device), adj.to(self.device), y.to(self.device)
            inner_optimizer.zero_grad()
            pred = self.inner_model(x, adj)
            loss = self.criterion(pred, y.long().flatten())
            loss.backward()
            inner_optimizer.step()


# ============================================================
# Helper: simple list dataset
# ============================================================
class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
