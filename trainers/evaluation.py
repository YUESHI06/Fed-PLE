"""
Evaluation utilities: compute Precision, Recall, F1, Accuracy.
No sklearn dependency - uses manual confusion matrix.
"""

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate CEGT model on test data.

    Returns:
        result_dict: dict with Accuracy, Precision, Recall, F1 score, FPR, FNR
    """
    all_predictions = []
    all_targets = []
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, adj, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            outputs = model(x, adj)
            labels_flat = labels.long().flatten()

            loss = criterion(outputs, labels_flat)
            total_loss += loss.item()

            preds = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
            all_predictions.extend(preds.cpu().tolist())
            all_targets.extend(labels_flat.cpu().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)

    # Compute metrics
    result_dict = compute_metrics(all_targets, all_predictions)
    result_dict['avg_loss'] = avg_loss
    return result_dict


def compute_metrics(targets, predictions):
    """Compute standard classification metrics without sklearn."""
    tp = tn = fp = fn = 0
    for t, p in zip(targets, predictions):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 score': f1,
        'FPR': fpr,
        'FNR': fnr,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }


def print_results(result_dict, prefix=''):
    """Print evaluation results."""
    print(f"{prefix}Accuracy:  {result_dict['Accuracy']:.4f}")
    print(f"{prefix}Precision: {result_dict['Precision']:.4f}")
    print(f"{prefix}Recall:    {result_dict['Recall']:.4f}")
    print(f"{prefix}F1 score:  {result_dict['F1 score']:.4f}")
    print(f"{prefix}FPR:       {result_dict.get('FPR', 0):.4f}")
    print(f"{prefix}FNR:       {result_dict.get('FNR', 0):.4f}")


def save_results(result_dict, result_dir, method, vul, noise_type, noise_rate):
    """Save evaluation results to JSON file."""
    rate_dir = 'diff' if noise_type == 'diff_noise' else str(noise_rate)
    result_path = Path(result_dir) / rate_dir / method
    result_path.mkdir(parents=True, exist_ok=True)

    if noise_type == 'fn_noise':
        fname = f'fn_{vul}_result.json'
    elif noise_type == 'diff_noise':
        fname = f'diff_noise_{vul}_result.json'
    elif noise_type == 'pure':
        fname = f'pure_{vul}_result.json'
    else:
        fname = f'{vul}_result.json'

    fpath = result_path / fname

    if fpath.exists():
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            data.append(result_dict)
    else:
        data = [result_dict]

    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {fpath}")
