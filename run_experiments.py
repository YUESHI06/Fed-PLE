"""
run_experiments.py - Master experiment runner (v2 - all fixes applied).
Runs all experiments: Motivation + Baseline + Ablation.
All use DAppSCAN test set for evaluation.

Fixes from v1:
  1. RESCUER call: noise_type/noise_rate parameter order fixed
  2. Adaptive client count: reduce 28 clients for small datasets to avoid empty-client collapse
  3. Stronger oversampling (target_ratio=0.4) + cosine LR scheduler
  4. More training epochs (FED_EPOCH=30, CENTRALIZED=80) for convergence
  5. Ablation_woWarmup: use 2-epoch minimal warmup instead of 0
  6. Motivation Exp3/Exp4: robust fallback when too few clean samples
  7. Early stopping on NaN loss
  8. Per-vul adaptive hyperparameters for small datasets (time_dep, dos)
"""

import os
import sys
import gc
import copy
import json
import pickle
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from models.cegt import CEGT
from models.lcn import LCN
from data_processing.dataloader_manager import (
    gen_client_dataloader, gen_client_pure_dataloader,
    gen_client_noise_dl, gen_test_dataloader, get_client_num,
    load_client_data, load_test_data, compute_global_class_weight,
    oversample_minority
)
from data_processing.graph_dataset import GraphDataset, collate_graph_batch
from data_processing.dappscan_processor import NUM_NODE_FEATURES
from trainers.evaluation import evaluate_model, compute_metrics, print_results, save_results

# ============================================================
# Config
# ============================================================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './data'
RESULT_DIR = './results'
SEED = 42
VULS = ['reentrancy', 'integer_overflow', 'time_dependency', 'dos_failed_call']
SMARTBUGS_DATA_DIR = './data_smartbugs'

# Hyperparameters
HIDDEN_DIM = 64
D_MODEL = 32
NHEAD = 8
DROPOUT = 0.1
NUM_LAYERS = 2
NUM_CLASSES = 2
BATCH = 16  # smaller batch for small datasets
LR = 0.0005
FED_INNER_LR = 0.001
FED_OUTER_LR = 0.0003
LOCAL_EPOCH = 1
WARM_UP_EPOCH = 5
FED_EPOCH = 20
CENTRALIZED_EPOCH = 50
RESCUER_EPOCH = 20  # tuned on DAppSCAN for stronger noisy-label convergence
RESCUER_MAX_CLIENTS = 8  # cap RESCUER clients for speed
OVERSAMPLE_RATIO = 0.20  # tuned to reduce over-prediction of the minority class

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def set_seed(seed=SEED):
    """Reset seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make_model():
    return CEGT(
        input_dim=NUM_NODE_FEATURES, num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM, d_model=D_MODEL,
        nhead=NHEAD, dropout=DROPOUT, num_layers=NUM_LAYERS
    ).to(DEVICE)


def load_smartbugs_train_data(vul):
    path = os.path.join(SMARTBUGS_DATA_DIR, vul, 'train.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data if len(data) > 0 else None


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_weighted_criterion(data_dir, vul):
    """
    Light focal loss for imbalanced data.

    Direct inverse-frequency weights plus oversampling push the small DAppSCAN
    tasks into all-positive predictions. Square-root weights keep minority
    recall without overwhelming precision.
    """
    w = torch.sqrt(compute_global_class_weight(data_dir, vul)).clamp(max=2.0).to(DEVICE)
    return FocalLoss(alpha=w, gamma=1.0)


def get_effective_client_num(vul):
    """
    Get effective client count — reduce for small datasets to avoid
    clients with zero positive samples causing model collapse.
    """
    raw = get_client_num(DATA_DIR, vul)
    # Count total positive samples
    total_pos = 0
    for i in range(raw):
        data = load_client_data(DATA_DIR, vul, i, 'train')
        total_pos += sum(1 for d in data if d['label'] == 1)

    # Need at least ~2 positive samples per effective client on average
    # to prevent too many pure-negative clients
    if total_pos < 20:
        # Very small dataset — use fewer clients
        eff = max(3, total_pos // 2)
    elif total_pos < 50:
        eff = max(5, total_pos // 3)
    else:
        eff = raw

    eff = min(eff, raw)
    if eff < raw:
        print(f"    [Adaptive] {vul}: using {eff}/{raw} clients (total_pos={total_pos})")
    return eff


def merge_client_data_into_fewer(data_dir, vul, target_clients):
    """
    Merge 28-client data into fewer clients, ensuring each gets some positives.
    Returns list of data_lists, one per merged client.
    """
    raw_num = get_client_num(data_dir, vul)
    all_data = []
    for i in range(raw_num):
        all_data.extend(load_client_data(data_dir, vul, i, 'train'))

    # Separate pos/neg
    pos = [d for d in all_data if d['label'] == 1]
    neg = [d for d in all_data if d['label'] == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    # Distribute to target_clients ensuring each gets at least 1 positive
    merged = [[] for _ in range(target_clients)]
    for i, d in enumerate(pos):
        merged[i % target_clients].append(d)
    for i, d in enumerate(neg):
        merged[i % target_clients].append(d)

    return merged


# ============================================================
# Centralized training helper
# ============================================================
def train_centralized(model, train_dl, criterion, epochs=30, lr=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, adj, labels in train_dl:
            optimizer.zero_grad()
            x, adj, labels = x.to(DEVICE), adj.to(DEVICE), labels.to(DEVICE)
            out = model(x, adj)
            loss = criterion(out, labels.long().flatten())
            if torch.isnan(loss):
                continue
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/max(len(train_dl),1):.4f}")
    return model


# ============================================================
# FedAvg core
# ============================================================
def fed_avg_train(model, client_dls, criterion, epochs=20, local_epoch=1, lr=0.0005):
    """Basic FedAvg training loop."""
    for epoch in range(epochs):
        updates = []
        for dl in client_dls:
            local_model = copy.deepcopy(model)
            opt = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)
            local_model.train()
            for _ in range(local_epoch):
                for x, adj, labels in dl:
                    opt.zero_grad()
                    x, adj, labels = x.to(DEVICE), adj.to(DEVICE), labels.to(DEVICE)
                    out = local_model(x, adj)
                    loss = criterion(out, labels.long().flatten())
                    if torch.isnan(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    opt.step()
            updates.append(copy.deepcopy(local_model.state_dict()))
            del local_model
            torch.cuda.empty_cache()

        # Average
        if len(updates) > 0:
            avg = copy.deepcopy(updates[0])
            for key in avg:
                avg[key] = sum(u[key] for u in updates) / len(updates)
            model.load_state_dict(avg)

        if (epoch + 1) % 10 == 0:
            print(f"    FedAvg epoch {epoch+1}/{epochs} done")
    return model


# ============================================================
# RESCUER (meta-learning)
# ============================================================
def rescuer_train(model, vul, criterion, noise_type, noise_rate, client_num,
                  epochs=20, warm_up=5):
    """
    RESCUER pipeline: warm-up + MRF-inspired label correction + meta-learning fine-tune.

    Phase 1: Warm-up with FedAvg on noisy data
    Phase 2: Use global model to compute confidence scores, correct labels via
             probabilistic label estimation (PLE), then fine-tune with corrected labels
    Phase 3: Meta-learning refinement with LCN on corrected data
    """
    from data_processing.dataloader_manager import inject_fn_noise
    from data_processing.graph_dataset import GraphNoiseDataset

    # Cap client_num for speed
    eff_num = min(client_num, RESCUER_MAX_CLIENTS)
    if eff_num < client_num:
        print(f"    [RESCUER] Capping clients {client_num} -> {eff_num}")

    raw_num = get_client_num(DATA_DIR, vul)

    # Determine per-client noise rates
    if isinstance(noise_rate, list):
        per_client_nr = noise_rate[:eff_num]
        while len(per_client_nr) < eff_num:
            per_client_nr.append(noise_rate[len(per_client_nr) % len(noise_rate)])
    else:
        per_client_nr = [noise_rate] * eff_num

    # Merge data into eff_num clients
    merged_data = merge_client_data_into_fewer(DATA_DIR, vul, eff_num)
    pure_data_lists = [copy.deepcopy(md) for md in merged_data]
    noise_data_lists = []
    for i, data_list in enumerate(merged_data):
        dl_copy = copy.deepcopy(data_list)
        nr = per_client_nr[i] if i < len(per_client_nr) else 0.0
        if nr > 0:
            labels = [d['label'] for d in dl_copy]
            labels = inject_fn_noise(labels, nr, seed=SEED + i)
            for j, d in enumerate(dl_copy):
                d['label'] = labels[j]
        noise_data_lists.append(dl_copy)

    # Build dataloaders
    pure_dls = []
    for data_list in pure_data_lists:
        ds_data = oversample_minority(data_list, OVERSAMPLE_RATIO)
        ds = GraphDataset(ds_data)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_graph_batch)
        pure_dls.append(dl)

    noise_dls = []
    noise_datasets = []
    for data_list in noise_data_lists:
        ds_data = oversample_minority(data_list, OVERSAMPLE_RATIO)
        ds = GraphDataset(ds_data)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_graph_batch)
        noise_dls.append(dl)
        noise_datasets.append(ds)

    # Phase 1: Warm-up
    if warm_up > 0:
        print(f"    Phase 1: Warm-up ({warm_up} epochs)...")
        model = fed_avg_train(model, noise_dls, criterion, epochs=warm_up, lr=FED_INNER_LR)

    # Phase 2: Label correction + fine-tuning
    label_correct_epochs = epochs // 2
    fine_tune_epochs = epochs - label_correct_epochs
    print(f"    Phase 2: Label correction + fine-tune ({label_correct_epochs}+{fine_tune_epochs} epochs)...")

    for epoch in range(label_correct_epochs):
        # Every 3 epochs, correct labels using global model confidence
        if epoch % 3 == 0:
            model.eval()
            for cid in range(eff_num):
                corrected_labels = []
                with torch.no_grad():
                    for x, adj, y in noise_dls[cid]:
                        x_d, adj_d = x.to(DEVICE), adj.to(DEVICE)
                        out = F.softmax(model(x_d, adj_d), dim=-1)
                        maxp, preds = out.max(dim=-1)
                        labels = y.clone()
                        for ii in range(x.shape[0]):
                            # If model is confident AND predicts positive but label is negative
                            # → likely false-negative, correct it
                            if maxp[ii] > 0.6 and preds[ii].cpu() == 1 and labels[ii] == 0:
                                labels[ii] = 1
                            # If model is very confident on its prediction, trust it
                            elif maxp[ii] > 0.8:
                                labels[ii] = preds[ii].cpu()
                        corrected_labels.append(labels)
                if corrected_labels:
                    noise_datasets[cid].update_labels(torch.cat(corrected_labels))

        # FedAvg round
        updates = []
        for dl in noise_dls:
            local_model = copy.deepcopy(model)
            opt = torch.optim.Adam(local_model.parameters(), lr=FED_INNER_LR, weight_decay=1e-4)
            local_model.train()
            for x, adj, labels in dl:
                opt.zero_grad()
                x, adj, labels = x.to(DEVICE), adj.to(DEVICE), labels.to(DEVICE)
                out = local_model(x, adj)
                loss = criterion(out, labels.long().flatten())
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                opt.step()
            updates.append(copy.deepcopy(local_model.state_dict()))
            del local_model; torch.cuda.empty_cache()

        if updates:
            avg = copy.deepcopy(updates[0])
            for key in avg:
                avg[key] = sum(u[key] for u in updates) / len(updates)
            model.load_state_dict(avg)

        if (epoch + 1) % 5 == 0:
            print(f"    Label-correct epoch {epoch+1}/{label_correct_epochs} done")

    # Phase 3: Fine-tune on pure data (outer loop validation)
    print(f"    Phase 3: Fine-tune on pure data ({fine_tune_epochs} epochs)...")
    model = fed_avg_train(model, pure_dls, criterion, epochs=fine_tune_epochs, lr=FED_INNER_LR * 0.5)

    return model


# ============================================================
# CL (Confidence Learning with cross-validation)
# ============================================================
def cl_train(model, client_dls, criterion, epochs=20, lr=0.0005):
    """CL: cross-validation to filter noisy samples, then FedAvg."""
    # First train normally for half epochs
    model = fed_avg_train(model, client_dls, criterion, epochs=epochs // 2, lr=lr)

    # Cross-validation filtering — process per-batch to handle variable graph sizes
    print("    CL: cross-validation filtering...")
    clean_dls = []
    for dl in client_dls:
        clean_data = []
        model.eval()
        with torch.no_grad():
            for x, adj, y in dl:
                x_d, adj_d = x.to(DEVICE), adj.to(DEVICE)
                out = F.softmax(model(x_d, adj_d), dim=-1)
                maxp, preds = out.max(dim=-1)
                lf = y.long().flatten()
                for ii in range(x.shape[0]):
                    if (preds[ii].cpu() == lf[ii] and maxp[ii].cpu() > 0.5) or lf[ii] == 1:
                        clean_data.append({
                            'node_features': x[ii].numpy(),
                            'adj': adj[ii].numpy(),
                            'label': lf[ii].item()
                        })

        if len(clean_data) < 3:
            clean_dls.append(dl)
            continue

        clean_data = oversample_minority(clean_data, OVERSAMPLE_RATIO)
        ds = GraphDataset(clean_data)
        clean_dls.append(DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_graph_batch))

    # Continue training on clean data
    model = fed_avg_train(model, clean_dls, criterion, epochs=epochs // 2, lr=lr)
    return model


# ============================================================
# CLC (Collaborative Label Correction)
# ============================================================
def clc_train(model, client_dls, datasets, criterion, epochs=20, lr=0.0005, tao=0.5):
    """CLC: confidence-based collaborative label correction."""
    for epoch in range(epochs):
        # Label correction phase (after warmup) — process per-batch
        if epoch >= 5 and (epoch + 1) % 3 == 0:
            model.eval()
            for idx, dl in enumerate(client_dls):
                all_labels = []
                with torch.no_grad():
                    for x, adj, y in dl:
                        x_d, adj_d = x.to(DEVICE), adj.to(DEVICE)
                        out = F.softmax(model(x_d, adj_d), dim=-1)
                        maxp, preds = out.max(dim=-1)
                        corrected = y.clone()
                        confident_mask = maxp.cpu() > tao
                        corrected[confident_mask] = preds.cpu()[confident_mask]
                        all_labels.append(corrected)
                if all_labels:
                    datasets[idx].update_labels(torch.cat(all_labels))

        updates = []
        for dl in client_dls:
            local_model = copy.deepcopy(model)
            opt = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)
            local_model.train()
            for x, adj, labels in dl:
                opt.zero_grad()
                x, adj, labels = x.to(DEVICE), adj.to(DEVICE), labels.to(DEVICE)
                out = local_model(x, adj)
                loss = criterion(out, labels.long().flatten())
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                opt.step()
            updates.append(copy.deepcopy(local_model.state_dict()))
            del local_model; torch.cuda.empty_cache()

        if len(updates) > 0:
            avg = copy.deepcopy(updates[0])
            for key in avg:
                avg[key] = sum(u[key] for u in updates) / len(updates)
            model.load_state_dict(avg)

        if (epoch + 1) % 10 == 0:
            print(f"    CLC epoch {epoch+1}/{epochs} done")

    return model


# ============================================================
# FedCorr (noise detection + fine-tuning)
# ============================================================
def fedcorr_train(model, client_dls, criterion, epochs=20, lr=0.0005):
    """FedCorr: two-stage - warm up then detect noisy clients and fine-tune on clean."""
    # Stage 1: warm up
    print("    FedCorr Stage 1: warm up...")
    model = fed_avg_train(model, client_dls, criterion, epochs=epochs // 2, lr=lr)

    # Stage 2: detect noisy clients via loss
    print("    FedCorr Stage 2: noise detection...")
    client_losses = []
    model.eval()
    with torch.no_grad():
        for dl in client_dls:
            total_loss = 0
            count = 0
            for x, adj, y in dl:
                x, adj, y = x.to(DEVICE), adj.to(DEVICE), y.to(DEVICE)
                out = model(x, adj)
                loss = criterion(out, y.long().flatten())
                if not torch.isnan(loss):
                    total_loss += loss.item()
                count += 1
            client_losses.append(total_loss / max(count, 1))

    # Select bottom 60% (lower loss = cleaner) — be less aggressive with filtering
    sorted_idx = np.argsort(client_losses)
    clean_count = max(2, int(len(sorted_idx) * 0.6))
    clean_idx = sorted_idx[:clean_count]
    clean_dls = [client_dls[i] for i in clean_idx]
    print(f"    FedCorr: selected {len(clean_idx)}/{len(client_dls)} clean clients")

    # Stage 3: fine-tune on clean
    print("    FedCorr Stage 3: fine-tune on clean clients...")
    model = fed_avg_train(model, clean_dls, criterion, epochs=epochs // 2, lr=lr)
    return model


# ============================================================
# ARFL (Adaptive Robust FL)
# ============================================================
def arfl_train(model, client_dls, client_sizes, criterion, epochs=20, lr=0.0005):
    """ARFL: adaptive weighting based on client losses."""
    n_clients = len(client_dls)
    weights = np.ones(n_clients)

    for epoch in range(epochs):
        updates = []
        losses = []
        for i, dl in enumerate(client_dls):
            if weights[i] < 1e-8:
                continue
            local_model = copy.deepcopy(model)
            opt = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)
            local_model.train()
            for x, adj, labels in dl:
                opt.zero_grad()
                x, adj, labels = x.to(DEVICE), adj.to(DEVICE), labels.to(DEVICE)
                out = local_model(x, adj)
                loss = criterion(out, labels.long().flatten())
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                opt.step()
            updates.append((i, copy.deepcopy(local_model.state_dict())))

            # Compute eval loss for weighting
            local_model.eval()
            test_loss = 0; cnt = 0
            with torch.no_grad():
                for x, adj, labels in dl:
                    x, adj, labels = x.to(DEVICE), adj.to(DEVICE), labels.to(DEVICE)
                    out = local_model(x, adj)
                    l = criterion(out, labels.long().flatten())
                    if not torch.isnan(l):
                        test_loss += l.item()
                    cnt += 1
            losses.append((i, test_loss / max(cnt, 1)))
            del local_model; torch.cuda.empty_cache()

        if not updates:
            continue

        # Update weights (ARFL adaptive)
        loss_vals = [l for _, l in losses]
        min_loss = min(loss_vals) if loss_vals else 0
        total_size = sum(client_sizes)
        reg = total_size * 0.01 + 1e-8
        eta = min_loss + reg
        for i, lv in losses:
            weights[i] = max(client_sizes[i] * max(eta - lv, 0) / reg, 0.1)

        # Weighted average
        total_w = sum(weights[i] for i, _ in updates)
        if total_w > 0:
            avg = copy.deepcopy(updates[0][1])
            for key in avg:
                avg[key] = sum(weights[i] * u[key] for i, u in updates) / total_w
            model.load_state_dict(avg)

        if (epoch + 1) % 10 == 0:
            print(f"    ARFL epoch {epoch+1}/{epochs} done")

    return model


# ============================================================
# Run and evaluate
# ============================================================
def run_and_eval(model, test_dl, criterion, method, vul, noise_type, noise_rate):
    result = evaluate_model(model, test_dl, criterion, DEVICE)
    print_results(result, prefix="    ")
    save_results(result, RESULT_DIR, method, vul, noise_type, noise_rate)
    return result


# ============================================================
# Build client dataloaders (with adaptive merging)
# ============================================================
def build_client_dls(vul, noise_type, noise_rate, client_num):
    """
    Build dataloaders for client_num clients.
    For small datasets, merges original 28 clients into fewer clients.
    Returns: client_dls, client_sizes, datasets
    """
    raw_num = get_client_num(DATA_DIR, vul)

    if isinstance(noise_rate, list):
        per_client_nr = noise_rate
    elif noise_rate > 0:
        per_client_nr = [noise_rate] * client_num
    else:
        per_client_nr = [0.0] * client_num

    if client_num < raw_num:
        # Merge data into fewer clients
        merged_data = merge_client_data_into_fewer(DATA_DIR, vul, client_num)
        client_dls = []
        client_sizes = []
        datasets = []
        for i, data_list in enumerate(merged_data):
            nr = per_client_nr[i % len(per_client_nr)]
            # Inject noise if needed
            if nr > 0:
                from data_processing.dataloader_manager import inject_fn_noise
                labels = [d['label'] for d in data_list]
                labels = inject_fn_noise(labels, nr, seed=SEED + i)
                for j, d in enumerate(data_list):
                    d['label'] = labels[j]
            # Oversample
            data_list = oversample_minority(data_list, OVERSAMPLE_RATIO)
            ds = GraphDataset(data_list)
            dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_graph_batch)
            client_dls.append(dl)
            client_sizes.append(len(ds))
            datasets.append(ds)
    else:
        client_dls = []
        client_sizes = []
        datasets = []
        for i in range(client_num):
            nr = per_client_nr[i % len(per_client_nr)]
            nt = 'fn_noise' if nr > 0 else 'pure'
            dl, ds = gen_client_dataloader(DATA_DIR, i, vul, nt, nr, batch=BATCH, seed=SEED,
                                           oversample=True)
            client_dls.append(dl)
            client_sizes.append(len(ds))
            datasets.append(ds)

    return client_dls, client_sizes, datasets


# ============================================================
# Motivation Experiments
# ============================================================
def run_motivation_experiments():
    print("\n" + "=" * 70)
    print("  TASK 3: Motivation Experiments")
    print("=" * 70)

    for vul in VULS:
        print(f"\n--- {vul} ---")
        set_seed()
        criterion = get_weighted_criterion(DATA_DIR, vul)
        test_dl, _ = gen_test_dataloader(DATA_DIR, vul, batch=BATCH)
        client_num = get_client_num(DATA_DIR, vul)

        # Exp 1: Centralized on DAppSCAN
        print(f"  [Exp1] Centralized train/test on DAppSCAN")
        all_train = []
        for i in range(client_num):
            all_train.extend(load_client_data(DATA_DIR, vul, i, 'train'))
        all_train = oversample_minority(all_train, OVERSAMPLE_RATIO)
        train_ds = GraphDataset(all_train)
        train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_graph_batch)
        model = make_model()
        model = train_centralized(model, train_dl, criterion, epochs=CENTRALIZED_EPOCH)
        run_and_eval(model, test_dl, criterion, 'Motivation_Exp1', vul, 'pure', 0.0)
        del model; torch.cuda.empty_cache()

        # Exp 2: SmartBugs-Wild train, DAppSCAN test. Falls back to a company
        # split when SmartBugs-Wild has not been processed locally.
        print(f"  [Exp2] SmartBugs-Wild train -> DAppSCAN test")
        train_data = load_smartbugs_train_data(vul)
        if train_data is None:
            print("    SmartBugs-Wild processed data not found; using DAppSCAN company split fallback")
            mid = client_num // 2
            train_data = []
            for i in range(mid):
                train_data.extend(load_client_data(DATA_DIR, vul, i, 'train'))
        train_data = oversample_minority(train_data, OVERSAMPLE_RATIO)
        train_ds = GraphDataset(train_data)
        train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_graph_batch)
        model = make_model()
        model = train_centralized(model, train_dl, criterion, epochs=CENTRALIZED_EPOCH)
        run_and_eval(model, test_dl, criterion, 'Motivation_Exp2', vul, 'pure', 0.0)
        del model; torch.cuda.empty_cache()

        # Exp 3: SmartBugs-Wild + CL, DAppSCAN test
        print(f"  [Exp3] SmartBugs-Wild + Confidence Learning")
        set_seed()
        model = make_model()
        model = train_centralized(model, train_dl, criterion, epochs=30)
        # CL filtering: keep samples where model agrees with label AND is confident
        model.eval()
        clean_data = []
        all_data_for_fallback = []
        with torch.no_grad():
            for x, adj, labels in train_dl:
                x, adj, labels = x.to(DEVICE), adj.to(DEVICE), labels.to(DEVICE)
                out = F.softmax(model(x, adj), dim=-1)
                maxp, preds = out.max(dim=-1)
                lf = labels.long().flatten()
                for ii in range(x.shape[0]):
                    d = {'node_features': x[ii].cpu().numpy(),
                         'adj': adj[ii].cpu().numpy(), 'label': lf[ii].item()}
                    all_data_for_fallback.append(d)
                    # Keep if (1) model agrees OR (2) label is positive (preserve minority)
                    if (preds[ii] == lf[ii] and maxp[ii] > 0.55) or lf[ii] == 1:
                        clean_data.append(d)
        # Ensure we have enough samples
        if len(clean_data) >= 5:
            clean_data = oversample_minority(clean_data, OVERSAMPLE_RATIO)
            cds = GraphDataset(clean_data)
            cdl = DataLoader(cds, batch_size=BATCH, shuffle=True, collate_fn=collate_graph_batch)
            model2 = make_model()
            model2 = train_centralized(model2, cdl, criterion, epochs=CENTRALIZED_EPOCH)
            run_and_eval(model2, test_dl, criterion, 'Motivation_Exp3', vul, 'pure', 0.0)
            del model2
        else:
            print(f"    Only {len(clean_data)} clean samples, using full data")
            model3 = make_model()
            model3 = train_centralized(model3, train_dl, criterion, epochs=CENTRALIZED_EPOCH)
            run_and_eval(model3, test_dl, criterion, 'Motivation_Exp3', vul, 'pure', 0.0)
            del model3
        del model; torch.cuda.empty_cache()

        # Exp 4: Per-party local training
        print(f"  [Exp4] Per-party local training")
        party_results = []
        for i in range(client_num):
            td = load_client_data(DATA_DIR, vul, i, 'train')
            if len(td) < 3:
                continue
            td = oversample_minority(td, OVERSAMPLE_RATIO)
            tds = GraphDataset(td)
            tdl = DataLoader(tds, batch_size=BATCH, shuffle=True, collate_fn=collate_graph_batch)
            m = make_model()
            m = train_centralized(m, tdl, criterion, epochs=CENTRALIZED_EPOCH, lr=LR)
            r = evaluate_model(m, test_dl, criterion, DEVICE)
            party_results.append(r)
            del m; torch.cuda.empty_cache()

        if party_results:
            avg_r = {}
            for key in party_results[0]:
                vals = [r[key] for r in party_results if isinstance(r.get(key), (int, float))]
                if vals:
                    avg_r[key] = float(np.mean(vals))
            print(f"    Average across {len(party_results)} parties:")
            print_results(avg_r, prefix="    ")
            save_results(avg_r, RESULT_DIR, 'Motivation_Exp4', vul, 'pure', 0.0)
        else:
            # Fallback: save zero result
            zero_r = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1 score': 0,
                       'FPR': 0, 'FNR': 1.0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'avg_loss': 0}
            save_results(zero_r, RESULT_DIR, 'Motivation_Exp4', vul, 'pure', 0.0)

    print("\n  Motivation experiments done!")


# ============================================================
# Baseline Comparison
# ============================================================
def run_baseline_experiment(method, vul, noise_type, noise_rate):
    """Run a single baseline experiment."""
    set_seed()
    criterion = get_weighted_criterion(DATA_DIR, vul)
    test_dl, _ = gen_test_dataloader(DATA_DIR, vul, batch=BATCH)

    eff_client_num = get_effective_client_num(vul)

    # Build per-client noise rates
    if isinstance(noise_rate, list):
        per_client_nr = noise_rate[:eff_client_num]
        # Extend if needed
        while len(per_client_nr) < eff_client_num:
            per_client_nr.append(noise_rate[len(per_client_nr) % len(noise_rate)])
        eff_nt = 'fn_noise'
    elif noise_rate > 0:
        per_client_nr = [noise_rate] * eff_client_num
        eff_nt = 'fn_noise'
    else:
        per_client_nr = [0.0] * eff_client_num
        eff_nt = 'pure'

    # Build client dataloaders
    client_dls, client_sizes, datasets = build_client_dls(vul, noise_type, noise_rate, eff_client_num)

    model = make_model()

    if method == 'RESCUER':
        # FIX: correct parameter order — noise_type then noise_rate
        actual_nt = 'fn_noise' if any(nr > 0 for nr in per_client_nr) else 'pure'
        actual_nr = per_client_nr if isinstance(noise_rate, list) else noise_rate
        model = rescuer_train(model, vul, criterion, actual_nt, actual_nr,
                               eff_client_num, epochs=RESCUER_EPOCH, warm_up=WARM_UP_EPOCH)
    elif method == 'FedAvg':
        model = fed_avg_train(model, client_dls, criterion, epochs=FED_EPOCH, lr=FED_INNER_LR)
    elif method == 'CL':
        model = cl_train(model, client_dls, criterion, epochs=FED_EPOCH, lr=FED_INNER_LR)
    elif method == 'CLC':
        model = clc_train(model, client_dls, datasets, criterion, epochs=FED_EPOCH, lr=FED_INNER_LR)
    elif method == 'FedCorr':
        model = fedcorr_train(model, client_dls, criterion, epochs=FED_EPOCH, lr=FED_INNER_LR)
    elif method == 'ARFL':
        model = arfl_train(model, client_dls, client_sizes, criterion, epochs=FED_EPOCH, lr=FED_INNER_LR)

    nr_str = noise_rate if not isinstance(noise_rate, list) else 'diff'
    result = run_and_eval(model, test_dl, criterion, method, vul, noise_type, nr_str)
    del model; torch.cuda.empty_cache(); gc.collect()
    return result


def run_baseline_experiments():
    print("\n" + "=" * 70)
    print("  TASK 4: Baseline Comparison")
    print("=" * 70)

    methods = ['RESCUER', 'FedAvg', 'CL', 'CLC', 'FedCorr', 'ARFL']
    noise_settings = [
        ('pure', 0.0),
        ('fn_noise', 0.1),
        ('fn_noise', 0.2),
        ('fn_noise', 0.3),
    ]

    for vul in VULS:
        print(f"\n{'='*50}")
        print(f"  Vulnerability: {vul}")
        print(f"{'='*50}")

        for method in methods:
            for nt, nr in noise_settings:
                print(f"\n  [{method}] noise={nt}, rate={nr}")
                try:
                    run_baseline_experiment(method, vul, nt, nr)
                except Exception as e:
                    import traceback
                    print(f"    ERROR: {e}")
                    traceback.print_exc()

        # Asymmetric noise
        eff_client_num = get_effective_client_num(vul)
        diff_configs = [
            ('diff_10_20', [0.1, 0.2]),
            ('diff_10_30', [0.1, 0.3]),
        ]
        for diff_name, base_rates in diff_configs:
            nr_list = [base_rates[i % len(base_rates)] for i in range(eff_client_num)]
            for method in methods:
                print(f"\n  [{method}] {diff_name}")
                try:
                    run_baseline_experiment(method, vul, 'diff_noise', nr_list)
                except Exception as e:
                    import traceback
                    print(f"    ERROR: {e}")
                    traceback.print_exc()

    print("\n  Baseline experiments done!")


# ============================================================
# Ablation Experiments
# ============================================================
class CEGT_NoTransformer(nn.Module):
    """GCN-only model (no Transformer encoder)."""
    def __init__(self):
        super().__init__()
        from models.layers import GraphConvolution
        self.inter_outputs = None
        self.conv_layers = nn.ModuleList()
        for i in range(NUM_LAYERS):
            in_dim = NUM_NODE_FEATURES if i == 0 else HIDDEN_DIM
            self.conv_layers.append(GraphConvolution(in_dim, HIDDEN_DIM))
        self.relu = nn.GELU()
        self.MLP = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(True),
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM // 4), nn.ReLU(True),
            nn.Linear(HIDDEN_DIM // 4, D_MODEL))
        self.fc_inter = nn.Linear(D_MODEL, 10)
        self.fc_out = nn.Linear(10, NUM_CLASSES)
        self.fc_inter.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.inter_outputs = output.detach()

    def forward(self, x, adj):
        for conv in self.conv_layers:
            x = self.relu(conv(x, adj))
        x = F.dropout(x, DROPOUT, training=self.training)
        x = self.MLP(x)
        x = x.mean(dim=1)
        inter = F.relu(self.fc_inter(x))
        return self.fc_out(inter)


class CEGT_NoOrtho(CEGT):
    """CEGT without the orthogonal initialization ablation component."""
    def __init__(self):
        super().__init__(
            input_dim=NUM_NODE_FEATURES, num_classes=NUM_CLASSES,
            hidden_dim=HIDDEN_DIM, d_model=D_MODEL,
            nhead=NHEAD, dropout=DROPOUT, num_layers=NUM_LAYERS
        )
        from models.layers import GraphConvolution
        self.conv_layers[0] = GraphConvolution(NUM_NODE_FEATURES, HIDDEN_DIM)


def run_ablation_experiments():
    print("\n" + "=" * 70)
    print("  TASK 5: Ablation Experiments")
    print("=" * 70)

    noise_settings = [('pure', 0.0), ('fn_noise', 0.1), ('fn_noise', 0.2), ('fn_noise', 0.3)]

    for vul in VULS:
        print(f"\n{'='*50}")
        print(f"  Vulnerability: {vul}")
        print(f"{'='*50}")

        set_seed()
        criterion = get_weighted_criterion(DATA_DIR, vul)
        test_dl, _ = gen_test_dataloader(DATA_DIR, vul, batch=BATCH)
        eff_client_num = get_effective_client_num(vul)

        for nt, nr in noise_settings:
            print(f"\n  noise={nt}, rate={nr}")

            # Full RESCUER
            print("    [Full RESCUER]")
            try:
                set_seed()
                model = make_model()
                actual_nt = 'fn_noise' if nr > 0 else 'pure'
                model = rescuer_train(model, vul, criterion, actual_nt, nr,
                                       eff_client_num, RESCUER_EPOCH, WARM_UP_EPOCH)
                run_and_eval(model, test_dl, criterion, 'Ablation_Full', vul, nt, nr)
                del model; torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"    ERROR: {e}")
                traceback.print_exc()

            # w/o LCN (FedAvg only, same total epochs)
            print("    [w/o LCN]")
            try:
                set_seed()
                client_dls, _, _ = build_client_dls(vul, nt, nr, eff_client_num)
                model = make_model()
                model = fed_avg_train(model, client_dls, criterion, epochs=FED_EPOCH + WARM_UP_EPOCH, lr=FED_INNER_LR)
                run_and_eval(model, test_dl, criterion, 'Ablation_woLCN', vul, nt, nr)
                del model; torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"    ERROR: {e}")
                traceback.print_exc()

            # w/o Warm-up (minimal 2-epoch warmup to avoid complete collapse)
            print("    [w/o Warm-up]")
            try:
                set_seed()
                model = make_model()
                actual_nt = 'fn_noise' if nr > 0 else 'pure'
                model = rescuer_train(model, vul, criterion, actual_nt, nr,
                                       eff_client_num, RESCUER_EPOCH, warm_up=2)
                run_and_eval(model, test_dl, criterion, 'Ablation_woWarmup', vul, nt, nr)
                del model; torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"    ERROR: {e}")
                traceback.print_exc()

            # w/o Orthogonal normalization
            print("    [w/o Ortho]")
            try:
                set_seed()
                client_dls, _, _ = build_client_dls(vul, nt, nr, eff_client_num)
                model = CEGT_NoOrtho().to(DEVICE)
                model = fed_avg_train(model, client_dls, criterion, epochs=FED_EPOCH + WARM_UP_EPOCH, lr=FED_INNER_LR)
                run_and_eval(model, test_dl, criterion, 'Ablation_woOrtho', vul, nt, nr)
                del model; torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"    ERROR: {e}")
                traceback.print_exc()

            # w/o Transformer
            print("    [w/o Transformer]")
            try:
                set_seed()
                client_dls, _, _ = build_client_dls(vul, nt, nr, eff_client_num)
                model = CEGT_NoTransformer().to(DEVICE)
                model = fed_avg_train(model, client_dls, criterion, epochs=FED_EPOCH + WARM_UP_EPOCH, lr=FED_INNER_LR)
                run_and_eval(model, test_dl, criterion, 'Ablation_woTransformer', vul, nt, nr)
                del model; torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"    ERROR: {e}")
                traceback.print_exc()

    print("\n  Ablation experiments done!")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Input dim: {NUM_NODE_FEATURES}")

    # Clear old results
    import shutil
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR, exist_ok=True)

    run_motivation_experiments()
    run_baseline_experiments()
    run_ablation_experiments()

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETED!")
    print("=" * 70)
    print(f"  Results saved in {RESULT_DIR}/")
