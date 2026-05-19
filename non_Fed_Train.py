"""
non_Fed_Train.py - Non-federated (centralized) training for Motivation experiments.

Supports:
  (1) Train/test on DAppSCAN (Motivation Exp 1)
  (2) Train on SmartBugs-Wild, test on DAppSCAN (Motivation Exp 2)
  (3) Train on SmartBugs-Wild + CL, test on DAppSCAN (Motivation Exp 3)
  (4) Per-party training on DAppSCAN, average metrics (Motivation Exp 4)
"""

import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from options import parse_args
from models.cegt import CEGT
from trainers.evaluation import evaluate_model, compute_metrics, print_results, save_results
from data_processing.dataloader_manager import (
    gen_client_dataloader, gen_client_pure_dataloader,
    gen_test_dataloader, get_client_num, load_client_data, load_test_data
)
from data_processing.graph_dataset import GraphDataset, collate_graph_batch
from data_processing.dappscan_processor import NUM_NODE_FEATURES
from torch.utils.data import DataLoader


def train_centralized(model, train_dl, criterion, device, epochs=50, lr=0.001):
    """Train model on centralized data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, adj, labels in train_dl:
            optimizer.zero_grad()
            x, adj, labels = x.to(device), adj.to(device), labels.to(device)
            outputs = model(x, adj)
            loss = criterion(outputs, labels.long().flatten())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}")
    return model


def load_smartbugs_train_data(args, vul):
    """Load processed SmartBugs-Wild train data for one vulnerability."""
    path = os.path.join(args.smartbugs_data_dir, vul, 'train.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data if len(data) > 0 else None


def exp1_dappscan_train_test(args, device):
    """Experiment 1: Train and test RESCUER on DAppSCAN for all 4 vulnerability types."""
    print("\n" + "="*60)
    print("Motivation Experiment 1: Train/Test on DAppSCAN")
    print("="*60)

    criterion = nn.CrossEntropyLoss()
    results = {}

    for vul in ['reentrancy', 'integer_overflow', 'time_dependency', 'dos_failed_call']:
        print(f"\n--- {vul} ---")
        client_num = get_client_num(args.data_dir, vul)
        if client_num == 0:
            print(f"  No data for {vul}, skipping")
            continue

        # Collect all client train data
        all_train = []
        for i in range(client_num):
            all_train.extend(load_client_data(args.data_dir, vul, i, 'train'))

        if len(all_train) < 5:
            print(f"  Too few samples ({len(all_train)}) for {vul}, skipping")
            continue

        train_ds = GraphDataset(all_train)
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_graph_batch)

        test_dl, _ = gen_test_dataloader(args.data_dir, vul, batch=args.batch)

        model = CEGT(
            input_dim=NUM_NODE_FEATURES, num_classes=args.num_classes,
            hidden_dim=args.hidden_dim, d_model=args.d_model,
            nhead=args.nhead, dropout=args.dropout, num_layers=args.num_layers
        ).to(device)

        model = train_centralized(model, train_dl, criterion, device, epochs=50)
        result = evaluate_model(model, test_dl, criterion, device)
        print_results(result, prefix="  ")
        results[vul] = result
        save_results(result, './results', 'Motivation_Exp1', vul, 'pure', 0.0)

    return results


def exp2_cross_dataset(args, device):
    """
    Experiment 2: Train on SmartBugs-Wild equivalent (using a subset of DAppSCAN
    as proxy since SmartBugs-Wild graphs aren't pre-processed), test on DAppSCAN.

    Note: We simulate cross-dataset by using half the companies for training
    and the other half for testing, to demonstrate domain shift.
    """
    print("\n" + "="*60)
    print("Motivation Experiment 2: Cross-dataset (domain shift simulation)")
    print("="*60)

    criterion = nn.CrossEntropyLoss()
    results = {}

    for vul in ['reentrancy', 'integer_overflow', 'time_dependency', 'dos_failed_call']:
        print(f"\n--- {vul} ---")
        client_num = get_client_num(args.data_dir, vul)
        if client_num < 2:
            print(f"  Need >=2 clients for cross-dataset exp, skipping {vul}")
            continue

        train_data = load_smartbugs_train_data(args, vul)
        test_dl, _ = gen_test_dataloader(args.data_dir, vul, batch=args.batch)
        if train_data is None:
            print("  SmartBugs-Wild processed data not found; using DAppSCAN company split fallback")
            mid = client_num // 2
            train_data = []
            for i in range(mid):
                train_data.extend(load_client_data(args.data_dir, vul, i, 'train'))

        if len(train_data) < 5:
            print(f"  Too few samples for {vul}, skipping")
            continue

        train_ds = GraphDataset(train_data)
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_graph_batch)

        model = CEGT(
            input_dim=NUM_NODE_FEATURES, num_classes=args.num_classes,
            hidden_dim=args.hidden_dim, d_model=args.d_model,
            nhead=args.nhead, dropout=args.dropout, num_layers=args.num_layers
        ).to(device)

        model = train_centralized(model, train_dl, criterion, device, epochs=50)
        result = evaluate_model(model, test_dl, criterion, device)
        print_results(result, prefix="  ")
        results[vul] = result
        save_results(result, './results', 'Motivation_Exp2', vul, 'pure', 0.0)

    return results


def exp3_cross_dataset_with_cl(args, device):
    """
    Experiment 3: Cross-dataset training with Confidence Learning.
    Same split as Exp2 but with CL noise filtering.
    """
    print("\n" + "="*60)
    print("Motivation Experiment 3: Cross-dataset + Confidence Learning")
    print("="*60)

    criterion = nn.CrossEntropyLoss()
    results = {}

    for vul in ['reentrancy', 'integer_overflow', 'time_dependency', 'dos_failed_call']:
        print(f"\n--- {vul} ---")
        client_num = get_client_num(args.data_dir, vul)
        if client_num < 2:
            print(f"  Need >=2 clients, skipping {vul}")
            continue

        train_data = load_smartbugs_train_data(args, vul)
        test_dl, _ = gen_test_dataloader(args.data_dir, vul, batch=args.batch)
        if train_data is None:
            print("  SmartBugs-Wild processed data not found; using DAppSCAN company split fallback")
            mid = client_num // 2
            train_data = []
            for i in range(mid):
                train_data.extend(load_client_data(args.data_dir, vul, i, 'train'))

        if len(train_data) < 10:
            print(f"  Too few samples for {vul}, skipping")
            continue

        # Phase 1: Train initial model
        train_ds = GraphDataset(train_data)
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_graph_batch)

        model = CEGT(
            input_dim=NUM_NODE_FEATURES, num_classes=args.num_classes,
            hidden_dim=args.hidden_dim, d_model=args.d_model,
            nhead=args.nhead, dropout=args.dropout, num_layers=args.num_layers
        ).to(device)

        model = train_centralized(model, train_dl, criterion, device, epochs=30)

        # Phase 2: Confidence Learning - filter noisy samples
        model.eval()
        clean_data = []
        with torch.no_grad():
            for x, adj, labels in train_dl:
                x, adj, labels = x.to(device), adj.to(device), labels.to(device)
                outputs = model(x, adj)
                probs = F.softmax(outputs, dim=-1)
                max_probs, preds = probs.max(dim=-1)
                labels_flat = labels.long().flatten()
                # Keep samples where prediction matches label with high confidence
                mask = (preds == labels_flat) & (max_probs > args.confidence_thres)
                for idx in mask.nonzero(as_tuple=False).squeeze(1):
                    i_val = idx.item()
                    clean_data.append({
                        'node_features': x[i_val].cpu().numpy(),
                        'adj': adj[i_val].cpu().numpy(),
                        'label': labels_flat[i_val].item()
                    })

        if len(clean_data) < 5:
            print(f"  Too few clean samples ({len(clean_data)}) after CL, using all data")
            clean_data = train_data

        # Phase 3: Retrain on clean data
        clean_ds = GraphDataset(clean_data)
        clean_dl = DataLoader(clean_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_graph_batch)

        model2 = CEGT(
            input_dim=NUM_NODE_FEATURES, num_classes=args.num_classes,
            hidden_dim=args.hidden_dim, d_model=args.d_model,
            nhead=args.nhead, dropout=args.dropout, num_layers=args.num_layers
        ).to(device)

        model2 = train_centralized(model2, clean_dl, criterion, device, epochs=50)

        result = evaluate_model(model2, test_dl, criterion, device)
        print_results(result, prefix="  ")
        results[vul] = result
        save_results(result, './results', 'Motivation_Exp3', vul, 'pure', 0.0)

    return results


def exp4_per_party(args, device):
    """
    Experiment 4: Train per-party on DAppSCAN, test individually, average metrics.
    Demonstrates why federated approach is needed.
    """
    print("\n" + "="*60)
    print("Motivation Experiment 4: Per-party training (local only)")
    print("="*60)

    criterion = nn.CrossEntropyLoss()
    results = {}

    for vul in ['reentrancy', 'integer_overflow', 'time_dependency', 'dos_failed_call']:
        print(f"\n--- {vul} ---")
        client_num = get_client_num(args.data_dir, vul)
        if client_num == 0:
            print(f"  No data for {vul}, skipping")
            continue

        test_dl, _ = gen_test_dataloader(args.data_dir, vul, batch=args.batch)

        party_results = []
        for i in range(client_num):
            train_data = load_client_data(args.data_dir, vul, i, 'train')
            if len(train_data) < 3:
                continue

            train_ds = GraphDataset(train_data)
            train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  collate_fn=collate_graph_batch)

            model = CEGT(
                input_dim=NUM_NODE_FEATURES, num_classes=args.num_classes,
                hidden_dim=args.hidden_dim, d_model=args.d_model,
                nhead=args.nhead, dropout=args.dropout, num_layers=args.num_layers
            ).to(device)

            model = train_centralized(model, train_dl, criterion, device, epochs=50, lr=0.001)
            result = evaluate_model(model, test_dl, criterion, device)
            party_results.append(result)
            print(f"  Client {i}: F1={result['F1 score']:.4f}, Acc={result['Accuracy']:.4f}")

        if party_results:
            # Average metrics across parties
            avg_result = {}
            for key in party_results[0]:
                if isinstance(party_results[0][key], (int, float)):
                    avg_result[key] = np.mean([r[key] for r in party_results])
            print(f"\n  Average across {len(party_results)} parties:")
            print_results(avg_result, prefix="  ")
            results[vul] = avg_result
            save_results(avg_result, './results', 'Motivation_Exp4', vul, 'pure', 0.0)

    return results


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Running all Motivation Experiments...")
    exp1_dappscan_train_test(args, device)
    exp2_cross_dataset(args, device)
    exp3_cross_dataset_with_cl(args, device)
    exp4_per_party(args, device)
    print("\nAll Motivation Experiments completed!")
