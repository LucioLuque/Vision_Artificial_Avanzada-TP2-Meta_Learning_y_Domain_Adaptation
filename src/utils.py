import random
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

from episode_sampler import EpisodeSampler

def deterministic(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def save_tsne_snapshot(model, images, labels, device, save_path, title, perplexity=30):
    model.eval()

    with torch.no_grad():
        embeddings = model.encoder(images.to(device)).detach().cpu().numpy()

    y = labels.detach().cpu().numpy()

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=y,
        cmap="tab10",
        s=10,
        alpha=0.8,
    )
    plt.colorbar(scatter, ticks=list(range(10)), label="Clase")
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor="white")
    plt.show()
    plt.close()

def accuracy_from_logits(logits, y_true):
    preds = logits.argmax(dim=1)
    return (preds == y_true).float().mean().item()

def run_epoch(model, sampler, optimizer, criterion, device, episodes, training = True):
    epoch_loss = 0.0
    epoch_support_acc = 0.0
    epoch_query_acc = 0.0

    desc = "Train" if training else "Val"
    episode_bar = tqdm(range(episodes), desc=desc, leave=False)

    for episode in episode_bar:
        support_images, support_labels, query_images, query_labels, _ = sampler.sample_episode()
        
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)

        if training:
            optimizer.zero_grad()

        query_logits, prototypes, support_embeddings, _ = model.forward_episode(support_images, support_labels, query_images)

        loss = criterion(query_logits, query_labels)

        if training:
            loss.backward()
            optimizer.step()

        query_acc = accuracy_from_logits(query_logits, query_labels)

        support_logits = model.euclidean_logits(support_embeddings, prototypes)
        support_acc = accuracy_from_logits(support_logits, support_labels)

        epoch_loss += loss.item()
        epoch_support_acc += support_acc
        epoch_query_acc += query_acc

        #Progress bar update
        current_mean_loss = epoch_loss / (episode + 1)
        current_mean_support_acc = epoch_support_acc / (episode + 1)
        current_mean_query_acc = epoch_query_acc / (episode + 1)

        episode_bar.set_postfix({
            "loss": f"{current_mean_loss:.4f}",
            "s_acc": f"{current_mean_support_acc:.4f}",
            "q_acc": f"{current_mean_query_acc:.4f}",
        })

    mean_loss = epoch_loss / episodes
    mean_support_acc = epoch_support_acc / episodes
    mean_query_acc = epoch_query_acc / episodes

    return mean_loss, mean_support_acc, mean_query_acc

def train_protonet(model, train_sampler, val_sampler, optimizer, criterion, 
                   device, epochs, episodes, tsne_images=None, tsne_labels=None,
                   tsne_dir="../images/tsne_protonet"):
    history = {
        "train_loss": [],
        "train_support_acc": [],
        "train_query_acc": [],
        "val_loss": [],
        "val_support_acc": [],
        "val_query_acc": [],
    }

    os.makedirs(tsne_dir, exist_ok=True)

    # Snapshot inicial: antes de entrenar
    if tsne_images is not None and tsne_labels is not None:
        save_tsne_snapshot(
            model=model,
            images=tsne_images,
            labels=tsne_labels,
            device=device,
            save_path=os.path.join(tsne_dir, "tsne_epoch_1.png"),
            title="t-SNE - 1",
        )

    mid_epoch = epochs // 2

    epoch_bar = tqdm(range(epochs), desc="Epochs")

    for epoch in epoch_bar:
        #TRAIN
        model.train()
        train_loss, train_support_acc, train_query_acc = run_epoch(model, train_sampler, optimizer, criterion, device, episodes, training=True)

        history["train_loss"].append(train_loss)
        history["train_support_acc"].append(train_support_acc)
        history["train_query_acc"].append(train_query_acc)

        #EVAL
        model.eval()
        with torch.no_grad():
            val_loss, val_support_acc, val_query_acc = run_epoch(model, val_sampler, None, criterion, device, episodes, training=False)
        
        history["val_loss"].append(val_loss)
        history["val_support_acc"].append(val_support_acc)
        history["val_query_acc"].append(val_query_acc)

        epoch_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_q_acc": f"{train_query_acc:.4f}",
            "val_q_acc": f"{val_query_acc:.4f}",
        })

        if (tsne_images is not None and tsne_labels is not None) and (epoch == mid_epoch - 1 or epoch == epochs - 1):
            save_tsne_snapshot(
                model=model,
                images=tsne_images,
                labels=tsne_labels,
                device=device,
                save_path=os.path.join(tsne_dir, f"tsne_epoch_{epoch+1}.png"),
                title=f"t-SNE - {epoch+1}",
            )

    return history

def eval_episode(model, sampler, device):
    model.eval()
    with torch.no_grad():
        support_images, support_labels, query_images, query_labels, _ = sampler.sample_episode()

        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)

        query_logits, _, _, _ = model.forward_episode(support_images, support_labels, query_images)
        pred_labels = torch.argmax(query_logits, dim=1)
        correct = (pred_labels == query_labels).sum().item()
        total = query_labels.size(0)
    return correct, total

def evaluate_domain(model, images, labels, n_way, k, q, episodes, device):
    sampler = EpisodeSampler(images=images, labels=labels, n_way=n_way, k_shot=k, q_query=q)

    total_correct = 0
    total_samples = 0

    for _ in range(episodes):
        correct, total = eval_episode(model, sampler, device)
        total_correct += correct
        total_samples += total

    return total_correct / total_samples

def evaluate_domains(model, test_data, n_way, ks, q, episodes, device):
    accuracies = {}
    for domain_name, (images, labels) in test_data.items():
        domain_accuracies = []
        for k in ks:
            accuracy = evaluate_domain(model=model, images=images, labels=labels,
                                       n_way=n_way, k=k, q=q, episodes=episodes, device=device)
            domain_accuracies.append(accuracy)
        accuracies[domain_name] = domain_accuracies
    return accuracies