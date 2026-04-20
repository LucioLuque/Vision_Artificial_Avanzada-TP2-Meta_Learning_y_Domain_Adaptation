import torch
import torch.nn as nn
from torch.func import functional_call
from collections import OrderedDict

class MAMLNet(nn.Module):
    def __init__(self, encoder, n_way, protomaml=False):
        super().__init__()
        self.encoder = encoder
        self.n_way = n_way
        self.classifier = nn.Linear(encoder.embedding_dim, n_way)
        self.protomaml = protomaml

    def forward(self, x, params=None):
        if params is None:
            embeddings = self.encoder(x)
            logits = self.classifier(embeddings)
            return logits
        
        #Functional call uses a specific set of parameters, instead of the models parameters.
        return functional_call(self, params, (x,)) 

    def get_parameters(self, support_x=None, support_y=None):
        if self.protomaml:
            return self.get_protomaml_parameters(support_x, support_y)
        return OrderedDict(self.named_parameters())
    
    def compute_prototypes(self, support_embeddings, support_labels):
        prototypes = []

        for c in range(self.n_way):
            class_mask = support_labels == c
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes, dim=0)   # [n_way, D]
        return prototypes
    
    def get_protomaml_parameters(self, support_x, support_y):
        params = OrderedDict(self.named_parameters())
        
        support_embeddings = self.encoder(support_x)

        prototypes = self.compute_prototypes(support_embeddings, support_y) # [n_way, D]

        proto_w = 2 * prototypes
        proto_b = - (prototypes ** 2).sum(dim=1)

        params["classifier.weight"] = proto_w
        params["classifier.bias"] = proto_b

        return params
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)
