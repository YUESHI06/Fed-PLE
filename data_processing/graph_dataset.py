import torch
import numpy as np
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    """
    Dataset for CEGT model. Each sample is a contract graph.

    Stores:
        - node_features: list of (N_i, F) arrays
        - adj_matrices: list of (N_i, N_i) arrays
        - labels: list of int
    """
    def __init__(self, data_list):
        """
        Args:
            data_list: list of dicts with keys 'node_features', 'adj', 'label'
        """
        self.node_features = [d['node_features'] for d in data_list]
        self.adj_matrices = [d['adj'] for d in data_list]
        self.labels = [d['label'] for d in data_list]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.node_features[index]).float(),
            torch.from_numpy(self.adj_matrices[index]).float(),
            self.labels[index]
        )

    def update_labels(self, new_labels):
        """Update labels (for noise injection or label correction)."""
        if isinstance(new_labels, torch.Tensor):
            self.labels = new_labels.cpu().numpy().tolist()
        elif isinstance(new_labels, np.ndarray):
            self.labels = new_labels.tolist()
        else:
            self.labels = list(new_labels)


class GraphNoiseDataset(Dataset):
    """
    Dataset with both noisy labels and global model predicted labels.
    Used by RESCUER (PLE) meta-learning.
    """
    def __init__(self, data_list, noise_labels, global_labels=None):
        self.node_features = [d['node_features'] for d in data_list]
        self.adj_matrices = [d['adj'] for d in data_list]
        self.noise_labels = noise_labels
        self.global_labels = global_labels

    def __len__(self):
        return len(self.noise_labels)

    def __getitem__(self, index):
        x = torch.from_numpy(self.node_features[index]).float()
        adj = torch.from_numpy(self.adj_matrices[index]).float()
        nl = self.noise_labels[index]
        if self.global_labels is not None:
            gl = self.global_labels[index]
            return x, adj, nl, gl
        return x, adj, nl

    def update_global_labels(self, global_labels):
        if isinstance(global_labels, torch.Tensor):
            self.global_labels = global_labels.cpu().tolist()
        else:
            self.global_labels = list(global_labels)


def collate_graph_batch(batch):
    """
    Collate function for variable-size graphs.
    Zero-pads node features and adjacency matrices to max size in batch.

    Returns:
        x: (B, N_max, F) padded node features
        adj: (B, N_max, N_max) padded adjacency matrices
        labels: (B,) or tuple with noise/global labels
    """
    B = len(batch)

    # Handle different return formats (with/without global labels)
    has_global = len(batch[0]) == 4

    N_nodes = [batch[b][0].shape[0] for b in range(B)]
    F_dim = batch[0][0].shape[1]
    N_max = max(N_nodes)

    x = torch.zeros(B, N_max, F_dim)
    adj = torch.zeros(B, N_max, N_max)

    for b in range(B):
        n = N_nodes[b]
        x[b, :n] = batch[b][0]
        adj[b, :n, :n] = batch[b][1]

    labels = torch.tensor([batch[b][2] for b in range(B)], dtype=torch.long)

    if has_global:
        global_labels = torch.tensor([batch[b][3] for b in range(B)], dtype=torch.long)
        return x, adj, labels, global_labels

    return x, adj, labels
