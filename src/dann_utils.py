from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils import select_fixed_subset, get_embeddings_labels, plot_tsnes_model

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            class_output, _ = model(images, lambda_p=0.0)
            preds = class_output.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0

def train_baseline(model, train_source_loader, val_source_loader, val_target_loader, optimizer, epochs, device):
    class_loss = nn.CrossEntropyLoss()

    history = {
        "train_class_loss": [],
        "val_source_acc": [],
        "val_target_acc": []
    }

    num_batches = len(train_source_loader)

    epoch_bar = tqdm(range(epochs), desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        total_class_loss = 0.0

        batch_bar = tqdm(train_source_loader, total=num_batches, desc="Batches", leave=False)
        for (source_data, source_labels) in batch_bar:
            source_data = source_data.to(device)
            source_labels = source_labels.to(device)

            optimizer.zero_grad()

            s_class_output, _ = model(source_data, lambda_p=0.0)

            cls_loss = class_loss(s_class_output, source_labels)

            cls_loss.backward()
            optimizer.step()

            total_class_loss += cls_loss.item()
            batch_bar.set_postfix({"cls_loss": f"{cls_loss.item():.4f}"})

        val_source_acc = evaluate(model, val_source_loader, device)
        val_target_acc = evaluate(model, val_target_loader, device)

        avg_loss = total_class_loss / num_batches

        history["train_class_loss"].append(avg_loss)
        history["val_source_acc"].append(val_source_acc)
        history["val_target_acc"].append(val_target_acc)
        
        epoch_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "val_source_acc": f"{val_source_acc:.4f}",
            "val_target_acc": f"{val_target_acc:.4f}"
        })
    return history

def train_dann(model, train_source_loader, val_source_loader, train_target_loader, val_target_loader,
               optimizer, epochs, device):
    class_loss = nn.CrossEntropyLoss()
    domain_loss = nn.BCEWithLogitsLoss()

    num_batches = min(len(train_source_loader), len(train_target_loader))
    total_steps = epochs * num_batches
    step = 0

    history = {
        "train_loss": [],
        "train_class_loss": [],
        "train_domain_loss": [],
        "val_source_acc": [],
        "val_target_acc": []
    }

    epoch_bar = tqdm(range(epochs), desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        total_class_loss = 0.0
        total_domain_loss = 0.0
        batch_bar = tqdm(zip(train_source_loader, train_target_loader), total=num_batches, desc="Batches", leave=False)
        for (source_data, source_labels), (target_data, _) in batch_bar:
            source_data = source_data.to(device)
            source_labels = source_labels.to(device)
            target_data = target_data.to(device)

            p = step / max(total_steps -1, 1)
            lambda_p = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            optimizer.zero_grad()

            s_class_output, s_domain_output = model(source_data, lambda_p=lambda_p)
            _, t_domain_output = model(target_data, lambda_p=lambda_p)

            cls_loss = class_loss(s_class_output, source_labels)

            domain_output = torch.cat([s_domain_output, t_domain_output], dim=0)
            domain_labels = torch.cat([torch.ones(source_data.size(0), 1), torch.zeros(target_data.size(0), 1)]).to(device)

            dom_loss = domain_loss(domain_output, domain_labels)

            loss = cls_loss + dom_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_class_loss += cls_loss.item()
            total_domain_loss += dom_loss.item()
            step += 1 
            batch_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls_loss": f"{cls_loss.item():.4f}",
                "dom_loss": f"{dom_loss.item():.4f}",
                "lambda_p": f"{lambda_p:.4f}"
            })

        val_source_acc = evaluate(model, val_source_loader, device)
        val_target_acc = evaluate(model, val_target_loader, device)

        avg_loss = total_loss / num_batches

        history["train_loss"].append(avg_loss)
        history["train_class_loss"].append(total_class_loss / num_batches)
        history["train_domain_loss"].append(total_domain_loss / num_batches)
        history["val_source_acc"].append(val_source_acc)
        history["val_target_acc"].append(val_target_acc)
        
        epoch_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "val_source_acc": f"{val_source_acc:.4f}",
            "val_target_acc": f"{val_target_acc:.4f}"
        })
    return history

def plot_tsnes(models, models_name, test_data, samples_per_class, device, seed=42):
    fixed_images_labels = {}

    for domain in test_data:
        
        images, labels = test_data[domain]
        fixed_images, fixed_labels = select_fixed_subset(
            images,
            labels,
            samples_per_class=samples_per_class,
            seed=seed
        )
        fixed_images_labels[domain] = (fixed_images, fixed_labels)

    
    all_embeddings, all_labels, all_domains, all_models = get_embeddings_labels(models, models_name, fixed_images_labels, device)
    
    domain_colors = {
        "Mnist": "tab:blue",
        "Mnist-M": "tab:orange",
    }
    paths = {
        0 : "../images/dann/baseline, tsne.png",
        1: "../images/dann/baseline_ft_tsne.png",
        2: "../images/dann/dann_tsne.png"
    }
    cmap = plt.cm.get_cmap("tab10", 10)
    for i, model in enumerate(models):
        plot_tsnes_model(all_embeddings[i], all_labels[i], all_domains[i], domain_colors, cmap, paths[i])