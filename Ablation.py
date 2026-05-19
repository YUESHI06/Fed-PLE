"""
Ablation.py - Ablation experiments for RESCUER with CEGT.

Ablates key components:
1. w/o LCN (no label correction network, just FedAvg with CEGT)
2. w/o Warm-up (skip warm-up phase)
3. w/o Meta-learning (use fixed label correction instead of meta-learning)
4. w/o Orthogonal normalization (remove Ortho_Trans from CEGT)
5. w/o Transformer (use GCN-only model, no Transformer encoder)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import parse_args
from models.cegt import CEGT
from models.lcn import LCN
from trainers.server import Server
from trainers.clients import RESCUER_Client, FedAvg_Client
from trainers.evaluation import evaluate_model, print_results, save_results
from data_processing.dataloader_manager import (
    gen_client_dataloader, gen_client_pure_dataloader,
    gen_client_noise_dl, gen_test_dataloader, get_client_num
)
from data_processing.dappscan_processor import NUM_NODE_FEATURES
from data_processing.graph_dataset import collate_graph_batch


class CEGT_NoOrtho(CEGT):
    """CEGT without orthogonal weight normalization."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # Re-initialize first conv layer without Ortho_Trans
        from models.layers import GraphConvolution
        self.conv_layers[0] = GraphConvolution(self.input_dim, self.hidden_dim)


class CEGT_NoTransformer(nn.Module):
    """GCN-only model (no Transformer encoder). For ablation."""
    def __init__(self, input_dim, num_classes=2, hidden_dim=64, d_model=32,
                 dropout=0.03, num_layers=2, **kwargs):
        super().__init__()
        self.inter_outputs = None
        from models.layers import GraphConvolution

        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.conv_layers.append(GraphConvolution(in_dim, hidden_dim))

        self.relu = nn.GELU()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, d_model)
        )
        self.fc_inter = nn.Linear(d_model, 10)
        self.fc_out = nn.Linear(10, num_classes)
        self.dropout_rate = dropout
        self.fc_inter.register_forward_hook(self._get_intermediate_outputs)

    def _get_intermediate_outputs(self, module, input, output):
        self.inter_outputs = output.detach()

    def forward(self, x, adj):
        for conv in self.conv_layers:
            x = self.relu(conv(x, adj))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.MLP(x)
        x = x.mean(dim=1)
        inter = F.relu(self.fc_inter(x))
        logits = self.fc_out(inter)
        return logits


def run_rescuer_full(args, device, vul, model_class=CEGT, skip_warmup=False):
    """Run full RESCUER pipeline. Returns evaluation result."""
    criterion = nn.CrossEntropyLoss()
    client_num = get_client_num(args.data_dir, vul)
    if client_num == 0:
        return None

    model_kwargs = dict(
        input_dim=NUM_NODE_FEATURES, num_classes=args.num_classes,
        hidden_dim=args.hidden_dim, d_model=args.d_model,
        dropout=args.dropout, num_layers=args.num_layers
    )
    if model_class == CEGT or model_class == CEGT_NoOrtho:
        model_kwargs['nhead'] = args.nhead

    global_model = model_class(**model_kwargs).to(device)
    server = Server(args, global_model, device, criterion)

    noise_dls = []
    pure_dls = []
    for i in range(client_num):
        ndl, _ = gen_client_dataloader(args.data_dir, i, vul, args.noise_type,
                                        args.noise_rate, batch=args.batch, seed=args.seed)
        noise_dls.append(ndl)
        pdl, _ = gen_client_pure_dataloader(args.data_dir, i, vul, batch=args.batch)
        pure_dls.append(pdl)

    clients = []
    for i in range(client_num):
        inner = copy.deepcopy(server.global_model)
        outer = LCN(in_channels=args.lcn_in_channels).to(device)
        client = RESCUER_Client(args, criterion, device, inner, outer, None, pure_dls[i])
        clients.append(client)

    # Warm-up
    if not skip_warmup:
        for we in range(args.warm_up_epoch):
            server.initialize_epoch_updates(we)
            for cid in range(client_num):
                clients[cid].inner_model = copy.deepcopy(server.global_model)
                clients[cid].warm_up()
                server.save_train_updates(
                    copy.deepcopy(clients[cid].get_inner_parameters()),
                    clients[cid].result['sample'], clients[cid].result
                )
            server.average_weights()

    # Main training
    for epoch in range(args.epoch):
        server.initialize_epoch_updates(epoch)
        for cid in range(client_num):
            clients[cid].inner_model = copy.deepcopy(server.global_model)
            # Generate global labels in the original sample order.
            pred_dl, _ = gen_client_dataloader(
                args.data_dir, cid, vul, args.noise_type, args.noise_rate,
                batch=args.batch, shuffle=False, seed=args.seed, oversample=False
            )
            gl = []
            with torch.no_grad():
                server.global_model.eval()
                for x, adj, _ in pred_dl:
                    x, adj = x.to(device), adj.to(device)
                    out = server.global_model(x, adj)
                    gl.append(torch.argmax(F.softmax(out, dim=-1), dim=-1).cpu())
            conc_gl = torch.cat(gl, dim=0)
            ndl, _ = gen_client_noise_dl(args.data_dir, cid, vul, args.noise_type,
                                          args.noise_rate, conc_gl, batch=args.batch, seed=args.seed)
            clients[cid].noise_dataloader = ndl
            clients[cid].meta_train()
            server.save_train_updates(
                copy.deepcopy(clients[cid].get_inner_parameters()),
                clients[cid].result['sample'], clients[cid].result
            )
        server.average_weights()

    test_dl, _ = gen_test_dataloader(args.data_dir, vul, batch=args.batch)
    return evaluate_model(server.global_model, test_dl, criterion, device)


def run_fedavg_only(args, device, vul):
    """Run FedAvg without LCN (ablation: w/o LCN)."""
    criterion = nn.CrossEntropyLoss()
    client_num = get_client_num(args.data_dir, vul)
    if client_num == 0:
        return None

    global_model = CEGT(
        input_dim=NUM_NODE_FEATURES, num_classes=args.num_classes,
        hidden_dim=args.hidden_dim, d_model=args.d_model,
        nhead=args.nhead, dropout=args.dropout, num_layers=args.num_layers
    ).to(device)
    server = Server(args, global_model, device, criterion)

    clients = []
    for i in range(client_num):
        dl, _ = gen_client_dataloader(args.data_dir, i, vul, args.noise_type,
                                       args.noise_rate, batch=args.batch, seed=args.seed)
        c = FedAvg_Client(args, criterion, copy.deepcopy(global_model), dl)
        clients.append(c)

    for epoch in range(args.epoch + args.warm_up_epoch):
        server.initialize_epoch_updates(epoch)
        for i, c in enumerate(clients):
            c.model = copy.deepcopy(server.global_model)
            c.train()
            server.save_train_updates(copy.deepcopy(c.get_parameters()),
                                       c.result['sample'], c.result)
        server.average_weights()

    test_dl, _ = gen_test_dataloader(args.data_dir, vul, batch=args.batch)
    return evaluate_model(server.global_model, test_dl, criterion, device)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vuls = ['reentrancy', 'integer_overflow', 'time_dependency', 'dos_failed_call']
    ablation_results = {}

    for vul in vuls:
        print(f"\n{'='*60}")
        print(f"Ablation experiments for: {vul}")
        print(f"{'='*60}")

        # Full RESCUER
        print("\n[1/5] Full RESCUER (CEGT + LCN + Warm-up + Meta-learning)")
        r = run_rescuer_full(args, device, vul, CEGT, skip_warmup=False)
        if r:
            print_results(r, "  ")
            save_results(r, './results', 'Ablation_Full', vul, args.noise_type, args.noise_rate)

        # w/o LCN
        print("\n[2/5] w/o LCN (FedAvg only)")
        r = run_fedavg_only(args, device, vul)
        if r:
            print_results(r, "  ")
            save_results(r, './results', 'Ablation_woLCN', vul, args.noise_type, args.noise_rate)

        # w/o Warm-up
        print("\n[3/5] w/o Warm-up")
        r = run_rescuer_full(args, device, vul, CEGT, skip_warmup=True)
        if r:
            print_results(r, "  ")
            save_results(r, './results', 'Ablation_woWarmup', vul, args.noise_type, args.noise_rate)

        # w/o Ortho normalization
        print("\n[4/5] w/o Orthogonal normalization")
        r = run_rescuer_full(args, device, vul, CEGT_NoOrtho, skip_warmup=False)
        if r:
            print_results(r, "  ")
            save_results(r, './results', 'Ablation_woOrtho', vul, args.noise_type, args.noise_rate)

        # w/o Transformer
        print("\n[5/5] w/o Transformer (GCN only)")
        r = run_rescuer_full(args, device, vul, CEGT_NoTransformer, skip_warmup=False)
        if r:
            print_results(r, "  ")
            save_results(r, './results', 'Ablation_woTransformer', vul, args.noise_type, args.noise_rate)

    print("\n\nAll ablation experiments completed!")
