"""
SmartBugs-Wild graph processor for CEGT experiments.

The raw SmartBugs/CBGRU data found online is not packaged in one stable layout, so
this parser is deliberately tolerant:
  - it recursively reads Solidity files;
  - it infers positive labels from vulnerability words in the file path;
  - for each target vulnerability, files from the other target folders become
    negative samples;
  - it saves train.pkl/test.pkl under data_smartbugs/<vulnerability>/.
"""

import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np

try:
    from .dappscan_processor import build_contract_graph
except ImportError:
    from dappscan_processor import build_contract_graph


TARGETS = {
    'reentrancy': ('reentrancy', 'reentry', 'reentrant'),
    'integer_overflow': ('integer_overflow', 'overflow', 'underflow', 'arithmetic'),
    'time_dependency': ('time_dependency', 'timestamp', 'time_manipulation', 'block_time'),
    'dos_failed_call': ('dos_failed_call', 'denial_of_service', 'failed_call', 'unchecked_low_level'),
}


def _path_text(path):
    return str(path).replace('\\', '/').lower()


def infer_vulnerabilities(path):
    text = _path_text(path)
    labels = set()
    for vul, aliases in TARGETS.items():
        if any(alias in text for alias in aliases):
            labels.add(vul)
    return labels


def collect_graphs(raw_dir):
    raw_root = Path(raw_dir)
    samples = []
    for sol_path in raw_root.rglob('*.sol'):
        labels = infer_vulnerabilities(sol_path)
        if not labels:
            continue
        try:
            source = sol_path.read_text(encoding='utf-8', errors='ignore')
            if len(source.strip()) < 50:
                continue
            node_features, adj = build_contract_graph(source)
        except Exception:
            continue
        samples.append({
            'sol_path': str(sol_path),
            'vul_labels': sorted(labels),
            'node_features': node_features,
            'adj': adj,
        })
    return samples


def save_pickle(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(items, f)


def process_smartbugs_dataset(raw_dir, output_dir, target_vul='all', test_ratio=0.2, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    samples = collect_graphs(raw_dir)
    if not samples:
        raise RuntimeError(
            f'No labelled Solidity files found under {raw_dir}. '
            'Point --smartbugs_dir to SmartBugs-Wild or CBGRU/train_data.'
        )

    targets = list(TARGETS) if target_vul == 'all' else [target_vul]
    for vul in targets:
        data = []
        for sample in samples:
            item = {
                'sol_path': sample['sol_path'],
                'label': 1 if vul in sample['vul_labels'] else 0,
                'node_features': sample['node_features'],
                'adj': sample['adj'],
            }
            data.append(item)

        pos = sum(1 for d in data if d['label'] == 1)
        neg = len(data) - pos
        if pos == 0 or neg == 0:
            print(f'[WARN] {vul}: skipped because pos={pos}, neg={neg}')
            continue

        random.shuffle(data)
        n_test = max(1, int(len(data) * test_ratio))
        test_data = data[:n_test]
        train_data = data[n_test:]
        vul_dir = os.path.join(output_dir, vul)
        save_pickle(train_data, os.path.join(vul_dir, 'train.pkl'))
        save_pickle(test_data, os.path.join(vul_dir, 'test.pkl'))
        print(f'{vul}: train={len(train_data)}, test={len(test_data)}, pos={pos}, neg={neg}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smartbugs_dir', default='../SmartBugs-Wild')
    parser.add_argument('--output_dir', default='./data_smartbugs')
    parser.add_argument('--vul', default='all', choices=['all'] + list(TARGETS))
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    process_smartbugs_dataset(
        args.smartbugs_dir, args.output_dir, args.vul, args.test_ratio, args.seed
    )
