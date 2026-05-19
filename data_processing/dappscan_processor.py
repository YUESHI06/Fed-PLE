"""
DAppSCAN Dataset Processor

Parses DAppSCAN JSON vulnerability labels, maps to target vulnerability types,
extracts Solidity source code, converts to contract-level graphs (adjacency matrix + node features),
and partitions data by audit company for federated learning.

Target vulnerabilities:
- IO: SWC-101 Integer Overflow and Underflow
- RE: SWC-107 Reentrancy
- TD: SWC-116 Block values as a proxy for time (Time Dependency)
- DFC: SWC-113 DoS with Failed Call
"""

import os
import re
import json
import random
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set


# SWC category mapping
VUL_MAP = {
    'SWC-101-Integer Overflow and Underflow': 'integer_overflow',
    'SWC-107-Reentrancy': 'reentrancy',
    'SWC-116-Block values as a proxy for time': 'time_dependency',
    'SWC-113-DoS with Failed Call': 'dos_failed_call',
}

# Solidity keywords for node type classification
SOLIDITY_KEYWORDS = {
    'function', 'modifier', 'event', 'constructor', 'fallback', 'receive',
    'mapping', 'struct', 'enum', 'contract', 'library', 'interface',
    'if', 'else', 'for', 'while', 'do', 'return', 'require', 'assert',
    'revert', 'emit', 'using', 'import', 'pragma', 'address', 'uint',
    'int', 'bool', 'string', 'bytes', 'public', 'private', 'internal',
    'external', 'view', 'pure', 'payable', 'storage', 'memory', 'calldata',
    'msg', 'block', 'tx', 'now', 'this', 'super', 'selfdestruct', 'transfer',
    'send', 'call', 'delegatecall', 'staticcall', 'abi', 'keccak256',
    'sha256', 'ecrecover', 'addmod', 'mulmod', 'balance', 'push', 'pop',
    'delete', 'new', 'type', 'true', 'false', 'wei', 'ether', 'gwei',
    'seconds', 'minutes', 'hours', 'days', 'weeks',
}

# Build keyword to index mapping for one-hot encoding
KEYWORD_LIST = sorted(SOLIDITY_KEYWORDS)
KEYWORD_TO_IDX = {kw: i for i, kw in enumerate(KEYWORD_LIST)}
NUM_NODE_FEATURES = len(KEYWORD_LIST) + 1  # +1 for "other" category


def parse_dappscan_json(swcsource_dir: str) -> Dict:
    """
    Parse all JSON files in SWCsource directory.

    Returns:
        {
            sol_file_path: {
                'audit_company': str,
                'project_dir': str,
                'vulnerabilities': {vul_type: [line_info, ...]},
            }
        }
    """
    contracts = {}
    for root, dirs, files in os.walk(swcsource_dir):
        for f in files:
            if not f.endswith('.json'):
                continue
            json_path = os.path.join(root, f)
            try:
                with open(json_path, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
            except Exception:
                continue

            file_path = data.get('filePath', '')
            parts = file_path.split('/')
            if len(parts) < 3:
                continue

            # Audit company is the prefix before '-' in the project dir name
            project_dir = parts[2]
            company = project_dir.split('-')[0] if '-' in project_dir else project_dir

            # Map sol path
            sol_rel_path = file_path  # relative to DAppSCAN-source

            vuls = {}
            for swc in data.get('SWCs', []):
                cat = swc.get('category', '')
                if cat in VUL_MAP:
                    vul_type = VUL_MAP[cat]
                    if vul_type not in vuls:
                        vuls[vul_type] = []
                    vuls[vul_type].append({
                        'function': swc.get('function', 'N/A'),
                        'line': swc.get('lineNumber', 'N/A')
                    })

            if sol_rel_path not in contracts:
                contracts[sol_rel_path] = {
                    'audit_company': company,
                    'project_dir': project_dir,
                    'vulnerabilities': vuls,
                    'json_path': json_path,
                }
            else:
                # Merge vulnerabilities
                for vt, vl in vuls.items():
                    if vt not in contracts[sol_rel_path]['vulnerabilities']:
                        contracts[sol_rel_path]['vulnerabilities'][vt] = []
                    contracts[sol_rel_path]['vulnerabilities'][vt].extend(vl)

    return contracts


def tokenize_solidity(source_code: str) -> List[str]:
    """Simple tokenizer for Solidity source code."""
    # Remove comments
    source_code = re.sub(r'//.*?\n', '\n', source_code)
    source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)
    # Tokenize: split on non-alphanumeric
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', source_code)
    return tokens


def extract_functions(source_code: str) -> List[dict]:
    """
    Extract function-level blocks from Solidity source code.
    Each function becomes a node in the contract graph.
    Returns list of {'name': str, 'body': str, 'start_line': int}
    """
    lines = source_code.split('\n')
    functions = []
    # Pattern for function/modifier/constructor/fallback/receive
    func_pattern = re.compile(
        r'^\s*(function\s+\w+|modifier\s+\w+|constructor|fallback|receive)\s*\('
    )
    i = 0
    while i < len(lines):
        match = func_pattern.match(lines[i])
        if match:
            # Find the function name
            line_str = lines[i].strip()
            if line_str.startswith('function'):
                name_match = re.search(r'function\s+(\w+)', line_str)
                name = name_match.group(1) if name_match else f'func_{i}'
            elif line_str.startswith('modifier'):
                name_match = re.search(r'modifier\s+(\w+)', line_str)
                name = name_match.group(1) if name_match else f'modifier_{i}'
            elif line_str.startswith('constructor'):
                name = 'constructor'
            elif line_str.startswith('fallback'):
                name = 'fallback'
            elif line_str.startswith('receive'):
                name = 'receive'
            else:
                name = f'block_{i}'

            # Find matching braces
            brace_count = 0
            start = i
            body_lines = []
            found_open = False
            while i < len(lines):
                for ch in lines[i]:
                    if ch == '{':
                        brace_count += 1
                        found_open = True
                    elif ch == '}':
                        brace_count -= 1
                body_lines.append(lines[i])
                i += 1
                if found_open and brace_count == 0:
                    break
            functions.append({
                'name': name,
                'body': '\n'.join(body_lines),
                'start_line': start + 1
            })
        else:
            i += 1

    # If no functions found, treat entire contract as one node
    if not functions:
        functions.append({
            'name': 'contract_body',
            'body': source_code,
            'start_line': 1
        })

    return functions


def compute_node_features(func_body: str) -> np.ndarray:
    """
    Compute node feature vector for a function body.
    Uses keyword frequency as features (one-hot style, but with counts).
    Returns: (NUM_NODE_FEATURES,) array
    """
    tokens = tokenize_solidity(func_body)
    features = np.zeros(NUM_NODE_FEATURES, dtype=np.float32)
    for token in tokens:
        if token in KEYWORD_TO_IDX:
            features[KEYWORD_TO_IDX[token]] += 1
        else:
            features[-1] += 1  # "other" category

    # Normalize
    total = features.sum()
    if total > 0:
        features = features / total
    return features


def build_contract_graph(source_code: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a contract-level graph from Solidity source code.

    Nodes = functions/modifiers/constructors
    Edges = function calls between nodes

    Returns:
        node_features: (N, NUM_NODE_FEATURES)
        adj_matrix: (N, N) adjacency matrix
    """
    functions = extract_functions(source_code)
    n = len(functions)

    # Node features
    node_features = np.zeros((n, NUM_NODE_FEATURES), dtype=np.float32)
    for i, func in enumerate(functions):
        node_features[i] = compute_node_features(func['body'])

    # Adjacency matrix: edge if function i calls function j
    adj = np.eye(n, dtype=np.float32)  # self-loops
    func_names = [f['name'] for f in functions]
    for i, func in enumerate(functions):
        body_tokens = set(tokenize_solidity(func['body']))
        for j, other_func in enumerate(functions):
            if i != j and other_func['name'] in body_tokens:
                adj[i][j] = 1.0
                adj[j][i] = 1.0  # undirected

    return node_features, adj


def process_dappscan_dataset(dappscan_source_dir: str, output_dir: str,
                              target_vul: str, min_contracts_per_party: int = 10):
    """
    Complete pipeline: parse DAppSCAN → build graphs → partition by audit company.

    Args:
        dappscan_source_dir: path to DAppSCAN-source/
        output_dir: where to save processed data
        target_vul: one of 'reentrancy', 'integer_overflow', 'time_dependency', 'dos_failed_call'
        min_contracts_per_party: minimum contracts to form a federated party
    """
    swcsource_dir = os.path.join(dappscan_source_dir, 'SWCsource')
    contracts_dir = os.path.join(dappscan_source_dir, 'contracts')

    print(f"Parsing DAppSCAN JSON files for {target_vul}...")
    all_contracts = parse_dappscan_json(swcsource_dir)

    # Collect all sol files with their vulnerability labels
    # A contract is positive (label=1) if it has the target vulnerability
    # A contract is negative (label=0) if it has been audited but no target vulnerability found
    company_data = defaultdict(list)  # company -> [(sol_path, label, node_features, adj)]

    # First, collect all .sol files that have been audited (appear in JSON)
    processed = set()
    for sol_rel_path, info in all_contracts.items():
        if sol_rel_path in processed:
            continue
        processed.add(sol_rel_path)

        company = info['audit_company']
        has_vul = 1 if target_vul in info['vulnerabilities'] else 0

        # Find the actual .sol file on disk
        # sol_rel_path format: "DAppSCAN-source/contracts/Company-Project/.../*.sol"
        sol_abs_path = os.path.join(os.path.dirname(dappscan_source_dir), sol_rel_path)
        if not os.path.exists(sol_abs_path):
            continue

        try:
            with open(sol_abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            if len(source_code.strip()) < 50:
                continue
            node_features, adj = build_contract_graph(source_code)
            if node_features.shape[0] < 1:
                continue
        except Exception:
            continue

        company_data[company].append({
            'sol_path': sol_rel_path,
            'label': has_vul,
            'node_features': node_features,
            'adj': adj,
        })

    # Also include contracts from same audit projects that don't appear in JSON (negative samples)
    # Walk through contracts/ directory matching audit projects
    for company in list(company_data.keys()):
        existing_sols = {d['sol_path'] for d in company_data[company]}
        company_contracts_dir = os.path.join(contracts_dir, f'{company}*')
        # Find matching directories
        import glob
        for proj_dir in glob.glob(os.path.join(contracts_dir, f'{company}*')):
            for root, dirs, files in os.walk(proj_dir):
                for f in files:
                    if not f.endswith('.sol'):
                        continue
                    full_path = os.path.join(root, f)
                    # Construct relative path
                    rel_path = os.path.relpath(full_path, os.path.dirname(dappscan_source_dir))
                    rel_path = rel_path.replace('\\', '/')
                    if rel_path.startswith('./'):
                        rel_path = rel_path[2:]

                    if rel_path in existing_sols:
                        continue

                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as fp:
                            source_code = fp.read()
                        if len(source_code.strip()) < 50:
                            continue
                        node_features, adj = build_contract_graph(source_code)
                        if node_features.shape[0] < 1:
                            continue
                    except Exception:
                        continue

                    company_data[company].append({
                        'sol_path': rel_path,
                        'label': 0,  # Not in JSON → no known vulnerability
                        'node_features': node_features,
                        'adj': adj,
                    })
                    existing_sols.add(rel_path)

    # Merge small companies
    print(f"\nCompany statistics for {target_vul}:")
    small_companies = []
    large_companies = []
    for company, data in sorted(company_data.items()):
        n_pos = sum(1 for d in data if d['label'] == 1)
        n_neg = sum(1 for d in data if d['label'] == 0)
        print(f"  {company}: {len(data)} contracts ({n_pos} positive, {n_neg} negative)")
        if len(data) < min_contracts_per_party:
            small_companies.append(company)
        else:
            large_companies.append(company)

    # Merge small companies into groups
    merged_data = {}
    for company in large_companies:
        merged_data[company] = company_data[company]

    if small_companies:
        merged = []
        for company in small_companies:
            merged.extend(company_data[company])
        if len(merged) >= min_contracts_per_party:
            merged_data['merged_small'] = merged
        else:
            # Add to the largest existing party
            if large_companies:
                largest = max(large_companies, key=lambda c: len(company_data[c]))
                merged_data[largest].extend(merged)

    # Create client splits
    parties = sorted(merged_data.keys())
    n_parties = len(parties)
    print(f"\nTotal federated parties: {n_parties}")

    # Split each party's data into train/test (80/20)
    vul_dir = os.path.join(output_dir, target_vul)
    os.makedirs(vul_dir, exist_ok=True)

    all_test_data = []

    for client_id, party in enumerate(parties):
        data = merged_data[party]
        random.shuffle(data)

        # Split
        n = len(data)
        n_test = max(1, int(n * 0.2))
        test_data = data[:n_test]
        train_data = data[n_test:]

        all_test_data.extend(test_data)

        # Save client data
        client_dir = os.path.join(vul_dir, f'client_{client_id}')
        os.makedirs(client_dir, exist_ok=True)

        save_graph_data(train_data, os.path.join(client_dir, 'train.pkl'))
        save_graph_data(test_data, os.path.join(client_dir, 'test.pkl'))

        # Save metadata
        meta = {
            'party_name': party,
            'n_train': len(train_data),
            'n_test': len(test_data),
            'n_pos_train': sum(1 for d in train_data if d['label'] == 1),
            'n_neg_train': sum(1 for d in train_data if d['label'] == 0),
            'n_pos_test': sum(1 for d in test_data if d['label'] == 1),
            'n_neg_test': sum(1 for d in test_data if d['label'] == 0),
        }
        with open(os.path.join(client_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  Client {client_id} ({party}): train={len(train_data)}, test={len(test_data)}")

    # Save global test set (all test data combined)
    save_graph_data(all_test_data, os.path.join(vul_dir, 'test_global.pkl'))

    # Save party mapping
    party_map = {i: p for i, p in enumerate(parties)}
    with open(os.path.join(vul_dir, 'party_map.json'), 'w') as f:
        json.dump(party_map, f, indent=2)

    print(f"\nGlobal test set: {len(all_test_data)} contracts")
    print(f"Data saved to {vul_dir}")

    return n_parties


def save_graph_data(data_list: List[dict], path: str):
    """Save list of {node_features, adj, label, sol_path} to pickle."""
    with open(path, 'wb') as f:
        pickle.dump(data_list, f)


def load_graph_data(path: str) -> List[dict]:
    """Load graph data from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dappscan_dir', type=str,
                        default='../DAppSCAN-main/DAppSCAN-source')
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--vul', type=str, default='all',
                        choices=['all', 'reentrancy', 'integer_overflow', 'time_dependency', 'dos_failed_call'])
    parser.add_argument('--min_contracts', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.vul == 'all':
        vuls = ['reentrancy', 'integer_overflow', 'time_dependency', 'dos_failed_call']
    else:
        vuls = [args.vul]

    for vul in vuls:
        print(f"\n{'='*60}")
        print(f"Processing vulnerability: {vul}")
        print(f"{'='*60}")
        process_dappscan_dataset(args.dappscan_dir, args.output_dir, vul, args.min_contracts)
