import random
import numpy as np
import torch

def deterministic(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy_from_logits(logits, y_true):
    preds = logits.argmax(dim=1)
    return (preds == y_true).float().mean().item()

def select_fixed_subset(images, labels, samples_per_class=100, seed = 42):
    g = torch.Generator().manual_seed(seed)

    selected_images = []
    selected_labels = []

    classes = torch.unique(labels)

    for cls in classes:
        cls_indices = torch.where(labels == cls)[0]
        perm = torch.randperm(len(cls_indices), generator=g)
        chosen = cls_indices[perm[:samples_per_class]]

        selected_images.append(images[chosen])
        selected_labels.append(labels[chosen])

    selected_images = torch.cat(selected_images, dim=0)
    selected_labels = torch.cat(selected_labels, dim=0)

    return selected_images, selected_labels