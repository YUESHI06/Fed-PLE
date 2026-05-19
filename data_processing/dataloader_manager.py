"""
Data loading and noise injection utilities for federated learning with CEGT.
Includes oversampling for imbalanced data.
"""

import os
import copy
import random
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from .graph_dataset import GraphDataset, GraphNoiseDataset, collate_graph_batch


def load_client_data(data_dir, vul, client_id, split='train'):
    """Load processed graph data for a client."""
    path = os.path.join(data_dir, vul, f'client_{client_id}', f'{split}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_test_data(data_dir, vul):
    """Load global test dataset."""
    path = os.path.join(data_dir, vul, 'test_global.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def inject_fn_noise(labels, noise_rate, seed=None):
    """
    Inject false-negative noise: flip positive labels (1->0) with probability noise_rate.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    noisy_labels = list(labels)
    for i in range(len(noisy_labels)):
        if noisy_labels[i] == 1 and rng.random() < noise_rate:
            noisy_labels[i] = 0
    return noisy_labels


def oversample_minority(data_list, target_ratio=0.3):
    """
    Oversample positive (minority) class to reach target_ratio of total.
    Adds Gaussian noise to duplicated node features for augmentation.
    """
    pos = [d for d in data_list if d['label'] == 1]
    neg = [d for d in data_list if d['label'] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return data_list

    target_pos = int(len(neg) * target_ratio / (1 - target_ratio))
    if target_pos <= len(pos):
        return data_list

    oversampled_pos = list(pos)  # Keep originals
    while len(oversampled_pos) < target_pos:
        for d in pos:
            if len(oversampled_pos) >= target_pos:
                break
            aug = copy.deepcopy(d)
            # Add small Gaussian noise to node features for diversity
            noise = np.random.normal(0, 0.05, aug['node_features'].shape).astype(np.float32)
            aug['node_features'] = np.clip(aug['node_features'] + noise, 0, None)
            oversampled_pos.append(aug)

    result = neg + oversampled_pos
    random.shuffle(result)
    return result


def compute_class_weight(data_list):
    """Compute class weight for weighted CrossEntropyLoss. Capped at max_ratio."""
    pos = sum(1 for d in data_list if d['label'] == 1)
    neg = sum(1 for d in data_list if d['label'] == 0)
    total = pos + neg
    if pos == 0 or neg == 0:
        return torch.tensor([1.0, 1.0])
    w0 = total / (2.0 * neg)
    w1 = total / (2.0 * pos)
    # Cap the ratio to prevent NaN
    max_w = 10.0
    w1 = min(w1, max_w)
    return torch.tensor([w0, w1], dtype=torch.float32)


def compute_global_class_weight(data_dir, vul):
    """Compute class weight from all client training data. Capped at 10."""
    client_num = get_client_num(data_dir, vul)
    total_pos = 0
    total_neg = 0
    for i in range(client_num):
        data = load_client_data(data_dir, vul, i, 'train')
        total_pos += sum(1 for d in data if d['label'] == 1)
        total_neg += sum(1 for d in data if d['label'] == 0)
    total = total_pos + total_neg
    if total_pos == 0 or total_neg == 0:
        return torch.tensor([1.0, 1.0])
    w0 = total / (2.0 * total_neg)
    w1 = total / (2.0 * total_pos)
    w1 = min(w1, 10.0)
    return torch.tensor([w0, w1], dtype=torch.float32)


def get_client_num(data_dir, vul):
    """Get number of clients for a vulnerability type."""
    vul_dir = os.path.join(data_dir, vul)
    count = 0
    while os.path.exists(os.path.join(vul_dir, f'client_{count}')):
        count += 1
    return count


def gen_client_dataloader(data_dir, client_id, vul, noise_type='pure',
                          noise_rate=0.0, batch=16, shuffle=True, seed=None,
                          oversample=True, target_ratio=0.3):
    """
    Generate a dataloader for a federated client.
    Applies oversampling to handle class imbalance.
    """
    data_list = load_client_data(data_dir, vul, client_id, 'train')
    labels = [d['label'] for d in data_list]

    if noise_type == 'fn_noise' and noise_rate > 0:
        labels = inject_fn_noise(labels, noise_rate,
                                 seed=(seed + client_id) if seed else None)

    # Update labels in data_list
    for i, d in enumerate(data_list):
        d['label'] = labels[i]

    # Oversample minority class
    if oversample:
        data_list = oversample_minority(data_list, target_ratio=target_ratio)

    dataset = GraphDataset(data_list)
    dl = DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                    collate_fn=collate_graph_batch)
    return dl, dataset


def gen_client_pure_dataloader(data_dir, client_id, vul, batch=16, oversample=True):
    """Generate dataloader with clean (pure) labels for a client."""
    data_list = load_client_data(data_dir, vul, client_id, 'train')
    if oversample:
        data_list = oversample_minority(data_list, target_ratio=0.3)
    dataset = GraphDataset(data_list)
    dl = DataLoader(dataset, batch_size=batch, shuffle=True,
                    collate_fn=collate_graph_batch)
    return dl, dataset


def gen_client_noise_dl(data_dir, client_id, vul, noise_type, noise_rate,
                        global_labels, batch=16, seed=None):
    """
    Generate dataloader with both noise labels and global model predicted labels.
    Used by RESCUER (PLE).
    Note: no oversampling here as global_labels correspond 1:1 with data.
    """
    data_list = load_client_data(data_dir, vul, client_id, 'train')
    labels = [d['label'] for d in data_list]

    if noise_type == 'fn_noise' and noise_rate > 0:
        noise_labels = inject_fn_noise(labels, noise_rate,
                                       seed=(seed + client_id) if seed else None)
    else:
        noise_labels = list(labels)

    # Truncate global_labels if needed
    n = len(data_list)
    if isinstance(global_labels, torch.Tensor):
        gl = global_labels[:n].tolist()
    else:
        gl = list(global_labels)[:n]

    dataset = GraphNoiseDataset(data_list, noise_labels, gl)
    dl = DataLoader(dataset, batch_size=batch, shuffle=True,
                    collate_fn=collate_graph_batch)
    return dl, dataset


def gen_test_dataloader(data_dir, vul, batch=16):
    """Generate test dataloader (global test set from DAppSCAN)."""
    data_list = load_test_data(data_dir, vul)
    dataset = GraphDataset(data_list)
    dl = DataLoader(dataset, batch_size=batch, shuffle=False,
                    collate_fn=collate_graph_batch)
    return dl, dataset


def gen_arfl_dl(data_dir, client_id, vul, noise_type, noise_rate, batch=16, seed=None):
    """Generate dataloader for ARFL training."""
    dl, dataset = gen_client_dataloader(
        data_dir, client_id, vul, noise_type, noise_rate,
        batch=batch, shuffle=False, seed=seed
    )
    return dl, len(dataset)


def gen_diff_noise_dataloaders(data_dir, vul, client_num, noise_rates,
                                batch=16, seed=None):
    dataloaders = []
    datasets = []
    for i in range(client_num):
        nr = noise_rates[i] if i < len(noise_rates) else 0.0
        dl, ds = gen_client_dataloader(
            data_dir, i, vul, 'fn_noise', nr, batch=batch, seed=seed
        )
        dataloaders.append(dl)
        datasets.append(ds)
    return dataloaders, datasets
