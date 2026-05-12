import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

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

def plot_accuracies_model(ks, accuracies, ax):
    ax.plot(ks, accuracies["Mnist"], marker="o", label="MNIST")
    ax.plot(ks, accuracies["Mnist-M"], marker="o", label="MNIST-M")
    ax.plot(ks, accuracies["Svhn"], marker="o", label="SVHN")

    ax.set_xlabel("K-shot", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xticks(ks)
    ax.legend(fontsize=14)
    ax.grid(True)
    return ax

def plot_accuracies_dataset(dataset, ks, accuracies, ax, models_name):

    for i, model in enumerate(models_name):
        ax.plot(ks, accuracies[i][dataset], marker="o", label=f"{model}")

    ax.set_xlabel("K-shot", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xticks(ks)
    ax.legend(fontsize=14)
    ax.grid(True)
    ax.set_title(f"{dataset}", fontsize=14)
    return ax

def plot_accuracies_all_models(ks, accuracies, models_name):
    fig, axes = plt.subplots(1, len(models_name), figsize=(15, 5), sharey=True)
    for i, dataset in enumerate(["Mnist", "Mnist-M", "Svhn"]):
        plot_accuracies_dataset(dataset, ks, accuracies, axes[i], models_name)
        if i != 0:
            axes[i].set_ylabel("")
    fig.tight_layout()
    path = "../images/compare_models/accuracies_ks.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def get_embeddings_labels(models, models_name, fixed_images_labels, device):

    all_embeddings = []
    all_labels = []
    all_domains = []
    all_models = []

    for model, model_name in zip(models, models_name):
        model.eval()
        embeddings_labels = {}

        with torch.no_grad():
            for domain in fixed_images_labels:
                images, labels = fixed_images_labels[domain]
                images = images.to(device)
                embeddings = model.encoder(images).cpu().numpy()
                labels = labels.cpu().numpy()
                embeddings_labels[domain] = (embeddings, labels)

        all_embeddings_model = []
        all_labels_model = []
        all_domains_model = []
        all_models_model = []

        for domain in embeddings_labels:
            embeddings, labels = embeddings_labels[domain]
            all_embeddings_model.append(embeddings)
            all_labels_model.append(labels)
            all_domains_model.extend([domain] * len(labels))
            all_models_model.extend([model_name] * len(labels))

        all_embeddings_model = np.concatenate(all_embeddings_model, axis=0)
        all_labels_model = np.concatenate(all_labels_model, axis=0)
        all_domains_model = np.array(all_domains_model)
        all_models_model = np.array(all_models_model)

        all_embeddings.append(all_embeddings_model)
        all_labels.append(all_labels_model)
        all_domains.append(all_domains_model)
        all_models.append(all_models_model)

    print(len(all_models))
    return all_embeddings, all_labels, all_domains, all_models

def plot_tsnes_model(embeddings, labels, domains, domain_colors, cmap, path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto" )
    tsne_embeddings = tsne.fit_transform(embeddings)

    #plot 2, domain color and class color
    fontsize = 14
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

    for domain in np.unique(domains):
        domain_mask = domains    == domain

        axes[0].scatter(
            tsne_embeddings[domain_mask, 0],
            tsne_embeddings[domain_mask, 1],
            color=domain_colors[domain],
            alpha=0.7,
            s=20,
            label=domain
        )

        #subplot 2: class color
        scatter = axes[1].scatter(
            tsne_embeddings[:, 0],
            tsne_embeddings[:, 1],
            c=labels,
            cmap=cmap,
            vmin=0,
            vmax=9,
            alpha=0.7,
            s=20
        )
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=fontsize)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=fontsize)
    axes[0].grid(True)
    axes[0].legend(fontsize=16)

    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=fontsize)
    axes[1].grid(True)

    cbar = fig.colorbar(scatter, ax=axes[1], ticks=list(range(10)))
    cbar.set_label("Class")
    fig.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()