import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class Ortho_Trans(nn.Module):
    """Orthogonal weight normalization via Newton's iteration"""
    def __init__(self, T=5, norm_groups=4):
        super(Ortho_Trans, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-4

    def matrix_power3(self, Input):
        B = torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps * eye
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)
        S = S.div(norm_S)
        B = [torch.Tensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            B[t + 1] = torch.baddbmm(B[t], self.matrix_power3(B[t]), S, beta=1.5, alpha=-0.5)
        W = B[self.T].matmul(Zc).div_(norm_S.sqrt())
        return W.view_as(weight)


class Attention(nn.Module):
    """Iterative refinement attention"""
    def __init__(self, input_dim, hidden_dim, num_iterations=3):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.u = nn.Parameter(torch.randn(hidden_dim))
        nn.init.orthogonal_(self.W)

    def forward(self, x):
        u_temp = torch.tanh(torch.matmul(x, self.W))
        u_temp = torch.matmul(u_temp, self.u)
        alpha = F.softmax(u_temp, dim=1).unsqueeze(-1)
        r = (alpha * x).sum(dim=1, keepdim=True)
        for i in range(self.num_iterations):
            beta = F.softmax(torch.matmul(x, r.transpose(1, 2)).squeeze(-1), dim=1)
            beta = beta.unsqueeze(-1)
            r = (beta * x).sum(dim=1, keepdim=True)
        return r


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads, num_iterations=3)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(x))
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class CEGT(nn.Module):
    """
    Connectivity-Enhanced GCN-Transformer (CEGT) model for smart contract
    vulnerability detection in federated learning setting.

    Input: (node_features: B×N×input_dim, adj: B×N×N)
    Output: (logits: B×num_classes)

    Also provides intermediate features (inter_outputs) for meta-learning (LCN).
    """
    def __init__(self, input_dim, num_classes=2, hidden_dim=64, d_model=32,
                 nhead=8, dropout=0.03, num_layers=2):
        super(CEGT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.dropout_rate = dropout
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.inter_outputs = None

        # GCN layers (all output hidden_dim; MLP maps to d_model)
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.conv_layers.append(GraphConvolution(in_dim, hidden_dim))

        # Orthogonal weight normalization for first layer
        self.weight_normalization = Ortho_Trans(T=5, norm_groups=4)
        if self.conv_layers[0].weight.shape[0] % 4 == 0:
            self.conv_layers[0].weight.data = self.weight_normalization(
                self.conv_layers[0].weight
            )

        # MLP between GCN and Transformer
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, d_model)
        )

        # Transformer Encoder
        self.encoder = Encoder(d_model, nhead, hidden_dim, num_layers, dropout)

        # Activation
        self.relu = nn.GELU()

        # Classifier: intermediate feature dim = 10, matches RESCUE LCN interface
        self.fc_inter = nn.Linear(d_model, 10)
        self.fc_out = nn.Linear(10, num_classes)

        # Register hook to capture intermediate outputs for LCN
        self.fc_inter.register_forward_hook(self._get_intermediate_outputs)

    def _get_intermediate_outputs(self, module, input, output):
        self.inter_outputs = output.detach()

    def forward(self, x, adj):
        """
        Args:
            x: (B, N, input_dim) - node features
            adj: (B, N, N) - adjacency matrices
        Returns:
            logits: (B, num_classes)
        """
        mask = (adj.diagonal(dim1=1, dim2=2) > 0).float().unsqueeze(-1)

        # GCN layers
        for conv in self.conv_layers:
            x = self.relu(conv(x, adj))
            x = x * mask
        x = F.dropout(x, self.dropout_rate, training=self.training)

        # MLP
        x = self.MLP(x)
        x = x * mask

        # Transformer encoder
        x = self.encoder(x)
        x = x * mask

        # Graph-level readout: masked mean pooling
        denom = mask.sum(dim=1).clamp(min=1.0)
        x = x.sum(dim=1) / denom

        # Classifier
        inter = F.relu(self.fc_inter(x))  # (B, 10) - intermediate for LCN
        logits = self.fc_out(inter)  # (B, num_classes)

        return logits
