"""
collect_results.py - Collect and format all experiment results into tables.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_all_results(result_dir):
    """Load all JSON results from the result directory."""
    results = defaultdict(lambda: defaultdict(dict))
    for root, dirs, files in os.walk(result_dir):
        for f in files:
            if not f.endswith('_result.json'):
                continue
            fpath = os.path.join(root, f)
            # Parse path: results/{noise_rate}/{method}/{fn_|diff_noise_|pure_}{vul}_result.json
            parts = Path(fpath).relative_to(result_dir).parts
            if len(parts) < 3:
                continue
            noise_rate = parts[0]
            method = parts[1]
            fname = parts[2]

            # Parse noise type and vul from filename
            vul = fname.replace('_result.json', '')
            noise_type = 'noise'
            if vul.startswith('fn_'):
                noise_type = 'fn_noise'
                vul = vul[3:]
            elif vul.startswith('diff_noise_'):
                noise_type = 'diff_noise'
                vul = vul[11:]
            elif vul.startswith('pure_'):
                noise_type = 'pure'
                vul = vul[5:]

            try:
                with open(fpath, 'r') as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    data = data[-1]  # Take last run
                key = f"{noise_type}_{noise_rate}"
                results[method][(vul, key)] = data
            except Exception:
                continue

    return results


def format_table(results, vuls, noise_settings, methods, metric='F1 score'):
    """Format results into a table."""
    header = ['Method'] + [f"{v}_{ns}" for v in vuls for ns in noise_settings]
    rows = [header]

    for method in methods:
        row = [method]
        for vul in vuls:
            for ns in noise_settings:
                key = (vul, ns)
                if key in results.get(method, {}):
                    val = results[method][key].get(metric, 0)
                    row.append(f"{val:.4f}")
                else:
                    row.append("-")
        rows.append(row)

    # Print table
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for row in rows:
        print(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default='./results')
    args = parser.parse_args()

    results = load_all_results(args.result_dir)

    vuls = ['reentrancy', 'integer_overflow', 'time_dependency', 'dos_failed_call']
    methods_baseline = ['RESCUER', 'FedAvg', 'CL', 'CLC', 'FedCorr', 'ARFL']
    methods_motivation = ['Motivation_Exp1', 'Motivation_Exp2', 'Motivation_Exp3', 'Motivation_Exp4']
    methods_ablation = ['Ablation_Full', 'Ablation_woLCN', 'Ablation_woWarmup',
                        'Ablation_woOrtho', 'Ablation_woTransformer']

    pure_settings = ['pure_0.0']
    fn_settings = ['fn_noise_0.1', 'fn_noise_0.2', 'fn_noise_0.3']
    all_settings = pure_settings + fn_settings
    diff_settings = ['diff_noise_diff']

    for metric in ['F1 score', 'Precision', 'Recall', 'Accuracy']:
        print(f"\n{'='*80}")
        print(f"  {metric}")
        print(f"{'='*80}")

        print("\n--- Motivation Experiments ---")
        format_table(results, vuls, pure_settings, methods_motivation, metric)

        print("\n--- Baseline Comparison (Pure + FN Noise) ---")
        format_table(results, vuls, all_settings, methods_baseline, metric)

        print("\n--- Baseline Comparison (Asymmetric FN Noise) ---")
        format_table(results, vuls, diff_settings, methods_baseline, metric)

        print("\n--- Ablation ---")
        format_table(results, vuls, all_settings, methods_ablation, metric)
