import torch
import torch.nn as nn

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
        self.to(device)