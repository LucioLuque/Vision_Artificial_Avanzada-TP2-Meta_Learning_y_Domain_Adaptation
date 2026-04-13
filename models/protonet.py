import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class ProtoNetEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, embedding_dim=64):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),   # 32 -> 16
            ConvBlock(hidden_dim, hidden_dim),    # 16 -> 8
            ConvBlock(hidden_dim, hidden_dim),    # 8 -> 4
            ConvBlock(hidden_dim, hidden_dim),    # 4 -> 2
        )

        self.projection = nn.Linear(hidden_dim * 2 * 2, embedding_dim)

    def forward(self, x):
        x = self.features(x)                 # [B, 64, 2, 2]
        x = x.flatten(start_dim=1)           # [B, 256]
        x = self.projection(x)               # [B, embedding_dim]
        return x


class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        prototypes = []

        for c in range(n_way):
            class_mask = support_labels == c
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes, dim=0)   # [n_way, D]
        return prototypes

    def euclidean_logits(self, query_embeddings, prototypes):
        # [N_query, 1, D] - [1, n_way, D] -> [N_query, n_way, D]
        distances = ((query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).sum(dim=2)
        logits = -distances
        return logits

    def forward_episode(self, support_x, support_y, query_x):
        n_way = len(torch.unique(support_y))

        support_embeddings = self.encoder(support_x)   # [N_support, D]
        query_embeddings = self.encoder(query_x)       # [N_query, D]

        prototypes = self.compute_prototypes(
            support_embeddings=support_embeddings,
            support_labels=support_y,
            n_way=n_way,
        )

        logits = self.euclidean_logits(
            query_embeddings=query_embeddings,
            prototypes=prototypes,
        )

        return logits, prototypes, support_embeddings, query_embeddings
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)