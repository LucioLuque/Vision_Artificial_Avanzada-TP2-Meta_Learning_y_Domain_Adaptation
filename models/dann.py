from torch.autograd import Function
import torch
import torch.nn as nn

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.lambda_p = lambda_p
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_p
        return output, None
    

class DANNModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        embedding_dim = encoder.embedding_dim

        self.task_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)   # 1 logit para BCEWithLogitsLoss
        )

    def forward(self, x, lambda_p=0.0):
        features = self.encoder(x)
        reverse_features = ReverseLayerF.apply(features, lambda_p)
        class_output = self.task_classifier(features)
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output