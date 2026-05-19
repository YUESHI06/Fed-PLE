import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LCN(nn.Module):
    """
    Label Correction Network (outer model for meta-learning in RESCUER).
    Takes intermediate features from CEGT (h_x) and pseudo-labels to produce
    corrected label distributions.

    Input:
        h_x: (B, 10) - intermediate features from CEGT fc_inter layer
        pse_labels: (B, 2, num_classes) - stacked one-hot of global + noise labels
    Output:
        corrected_labels: (B, 1, num_classes)
    """
    def __init__(self, in_channels=10, hidden_dim=64, n_class=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.label_embedding = nn.Linear(n_class * 2, 128)
        self.fc1 = nn.Linear(in_channels + 128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_class)
        self.rev_temp1 = 1.0
        self.rev_temp2 = math.sqrt(2)

    def forward(self, h_x, pse_labels):
        lab_emb = self.label_embedding(pse_labels.flatten(start_dim=1))
        x = torch.cat((h_x, lab_emb), dim=1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.relu(self.fc3(x))
        x = x * self.rev_temp1
        x = F.softmax(x, dim=1)
        x = x * self.rev_temp2
        x = torch.bmm(x.unsqueeze(dim=1), pse_labels)
        return x
