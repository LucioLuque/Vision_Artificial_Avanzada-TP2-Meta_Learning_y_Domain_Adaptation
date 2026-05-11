from episode_sampler import EpisodeSampler
import torch
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

from utils import accuracy_from_logits

def inner_loop(model, support_images, support_labels, inner_updates, alpha, create_graph):
    params = model.get_parameters(support_images, support_labels)
    for step in range(inner_updates):
        support_logits = model(support_images, params)
        support_loss = F.cross_entropy(support_logits, support_labels)
        
        grads = torch.autograd.grad(support_loss, list(params.values()), create_graph=create_graph)
        params = OrderedDict((name, param - alpha * grad) for ((name, param), grad) in zip(params.items(), grads))

    return params

def run_episode(model, sampler, inner_updates, alpha, device, train):
    support_images, support_labels, query_images, query_labels, _ = sampler.sample_episode()
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    query_labels = query_labels.to(device)
    
    init_params = model.get_parameters(support_images, support_labels)

    support_logits_pre = model(support_images, init_params)
    support_acc_pre = accuracy_from_logits(support_logits_pre, support_labels)

    params = inner_loop(model, support_images, support_labels, inner_updates, alpha, create_graph=train)

    support_logits_post = model(support_images, params)
    support_acc_post = accuracy_from_logits(support_logits_post, support_labels)

    query_logits = model(query_images, params)
    query_loss = F.cross_entropy(query_logits, query_labels)
    query_acc_post = accuracy_from_logits(query_logits, query_labels)

    return query_loss, support_acc_pre, support_acc_post, query_acc_post

def run_meta_batch(model, sampler, optimizer, meta_batch_size, inner_updates, alpha, device, train=True):
    query_loss_sum = 0.0
    support_acc_pre_sum = 0.0
    support_acc_post_sum = 0.0
    query_acc_post_sum = 0.0
    meta_loss = 0.0
    
    for _ in range(meta_batch_size):
        query_loss, support_acc_pre, support_acc_post, query_acc_post = run_episode(model, sampler, inner_updates, alpha, device, train)
        meta_loss += query_loss
        query_loss_sum += query_loss.item()
        support_acc_pre_sum += support_acc_pre
        support_acc_post_sum += support_acc_post
        query_acc_post_sum += query_acc_post

    if train:
        meta_loss /= meta_batch_size
        optimizer.zero_grad()
        meta_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return query_loss_sum/meta_batch_size, support_acc_pre_sum/meta_batch_size, support_acc_post_sum/meta_batch_size, query_acc_post_sum/meta_batch_size

def train_maml(model, train_sampler, val_sampler, optimizer, meta_iterations, meta_batch_size, inner_updates, alpha, device):
    history = {
        "train_query_loss": [],
        "train_support_acc_pre": [],
        "train_support_acc_post": [],
        "train_query_acc_post": [],
        "val_query_loss": [],
        "val_support_acc_pre": [],
        "val_support_acc_post": [],
        "val_query_acc_post": [],
    }
    meta_iter_bar = tqdm(range(meta_iterations), desc="Meta-Iterations")
    for _ in meta_iter_bar:
        model.train()
        query_train_loss, support_train_acc, support_train_acc_post, query_train_acc_post = run_meta_batch(model, train_sampler, optimizer, meta_batch_size, inner_updates, alpha, device, train=True)
        model.eval()
        query_val_loss, support_val_acc, support_val_acc_post, query_val_acc_post = run_meta_batch(model, val_sampler, None, meta_batch_size, inner_updates, alpha, device, train=False)

        history["train_query_loss"].append(query_train_loss)
        history["train_support_acc_pre"].append(support_train_acc)
        history["train_support_acc_post"].append(support_train_acc_post)
        history["train_query_acc_post"].append(query_train_acc_post)
        history["val_query_loss"].append(query_val_loss)
        history["val_support_acc_pre"].append(support_val_acc)
        history["val_support_acc_post"].append(support_val_acc_post)
        history["val_query_acc_post"].append(query_val_acc_post)

        meta_iter_bar.set_postfix({
            "train_q_loss": f"{query_train_loss:.4f}",
            "train_q_acc_post": f"{query_train_acc_post:.4f}",
            "val_q_acc_post": f"{query_val_acc_post:.4f}",
        })

    return history

def eval_episode(model, sampler, inner_updates, alpha, device):
    model.eval()
    
    support_images, support_labels, query_images, query_labels, _ = sampler.sample_episode()

    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    query_labels = query_labels.to(device)

    params = inner_loop(model, support_images, support_labels, inner_updates, alpha, create_graph=False)
    
    query_logits = model(query_images, params)
    preds = torch.argmax(query_logits, dim=1)
    correct = (preds == query_labels).sum().item()
    total = query_labels.size(0)
    return correct, total

def evaluate_domain(model, images, labels, n_way, k, q, episodes, inner_updates, alpha, device):
    sampler = EpisodeSampler(images=images, labels=labels, n_way=n_way, k_shot=k, q_query=q)

    total_correct = 0
    total_samples = 0

    for _ in range(episodes):
        correct, total = eval_episode(model, sampler, inner_updates, alpha, device)
        total_correct += correct
        total_samples += total

    return total_correct / total_samples

def evaluate_domains_maml(model, test_data, n_way, ks, q, episodes, inner_updates, alpha, device):
    accuracies = {}
    for domain_name, (images, labels) in test_data.items():
        domain_accuracies = []
        for k in ks:
            accuracy = evaluate_domain(model=model, images=images, labels=labels,
                                       n_way=n_way, k=k, q=q, episodes=episodes, 
                                       inner_updates=inner_updates, alpha=alpha, 
                                       device=device)
            domain_accuracies.append(accuracy)
        accuracies[domain_name] = domain_accuracies
    return accuracies

def plot_maml_history(history, path):
    epochs = range(1, len(history["train_query_loss"]) + 1)
    fontsize = 14
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(epochs, history["train_support_acc_pre"], label="Train Support Acc Pre")
    axes[0].plot(epochs, history["train_support_acc_post"], label="Train Support Acc Post")
    axes[0].plot(epochs, history["val_support_acc_pre"], label="Val Support Acc Pre")
    axes[0].plot(epochs, history["val_support_acc_post"], label="Val Support Acc Post")
    axes[0].set_xlabel("Epoch", fontsize=fontsize)
    axes[0].set_ylabel("Accuracy", fontsize=fontsize)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=fontsize)
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_query_loss"], label="Train Query Loss")
    axes[1].plot(epochs, history["val_query_loss"], label="Val Query Loss")
    axes[1].set_xlabel("Epoch", fontsize=fontsize)
    axes[1].set_ylabel("Loss", fontsize=fontsize)
    axes[1].legend(fontsize=fontsize)
    axes[1].grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()