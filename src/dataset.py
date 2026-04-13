from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def save_splits(train_images, train_labels, val_images, val_labels, test_images, test_labels, output_path, config):
    torch.save({"images": train_images, "labels": train_labels}, output_path / "train.pt")
    torch.save({"images": val_images, "labels": val_labels}, output_path / "val.pt")
    torch.save({"images": test_images, "labels": test_labels}, output_path / "test.pt")
    torch.save(config, output_path / "config.pt")

def save_mnist_dataloaders(batch_size=32, num_workers=0, val_size=0.1):
    output_dir = "../dataset/mnist"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root=output_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=output_dir,
        train=False,
        download=True,
        transform=transform
    )

    val_size = int(val_size * len(train_dataset))
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_images = train_dataset.dataset.data[train_dataset.indices].clone()
    train_labels = train_dataset.dataset.targets[train_dataset.indices].clone()
    val_images = val_dataset.dataset.data[val_dataset.indices].clone()
    val_labels = val_dataset.dataset.targets[val_dataset.indices].clone()
    test_images = test_dataset.data.clone()
    test_labels = test_dataset.targets.clone()

    save_splits(
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels,
        output_path,
        {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "val_size": val_size,
            "mean": (0.1307,),
            "std": (0.3081,),
        },
    )

def save_svhn_dataloaders(batch_size=32, num_workers=0, val_size=0.1, output_dir="../dataset/svhn"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    
    train_dataset = datasets.SVHN(
        root=output_dir,
        split="train",
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.SVHN(
        root=output_dir,
        split="test",
        download=True,
        transform=transform
    )

    #creo que validation no es necesario!

    val_size = int(val_size * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_images = torch.from_numpy(train_dataset.dataset.data[train_dataset.indices]).clone()
    train_labels = torch.from_numpy(train_dataset.dataset.labels[train_dataset.indices]).long().clone()
    val_images = torch.from_numpy(val_dataset.dataset.data[val_dataset.indices]).clone()
    val_labels = torch.from_numpy(val_dataset.dataset.labels[val_dataset.indices]).long().clone()
    test_images = torch.from_numpy(test_dataset.data).clone()
    test_labels = torch.from_numpy(test_dataset.labels).long().clone()

    save_splits(
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels,
        output_path,
        {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "val_size": val_size,
            "mean": (0.4377, 0.4438, 0.4728),
            "std": (0.1980, 0.2010, 0.1970),
        },
    )

def read_mnist_m_labels(labels_file):
    items = []

    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Línea inválida en {labels_file}: '{line}'. "
                    "Esperaba: <filename> <label>"
                )

            filename, label = parts[0], int(parts[1])
            items.append((filename, label))

    return items

def load_mnist_m_split(images_dir, labels_file):
    items = read_mnist_m_labels(labels_file)

    images = []
    labels = []

    for filename, label in items:
        img_path = Path(images_dir) / filename
        if not img_path.exists():
            raise FileNotFoundError(f"No encontré la imagen: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.uint8)   # [H, W, C]

        images.append(torch.from_numpy(img))
        labels.append(label)

    images = torch.stack(images, dim=0).contiguous()   # [N, H, W, C]
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels

def compute_channel_stats_hwc_uint8(images):
    x = images.float() / 255.0
    mean = x.mean(dim=(0, 1, 2))
    std = x.std(dim=(0, 1, 2))

    std = torch.clamp(std, min=1e-8)

    return tuple(mean.tolist()), tuple(std.tolist())

def save_mnist_m_dataloaders(batch_size=32, num_workers=0, val_size=0.1):
    output_dir = "../dataset/mnist_m"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_images_dir = output_path / "mnist_m_train"
    test_images_dir = output_path / "mnist_m_test"
    train_labels_file = output_path / "mnist_m_train_labels.txt"
    test_labels_file = output_path / "mnist_m_test_labels.txt"

    full_train_images, full_train_labels = load_mnist_m_split(train_images_dir, train_labels_file)
    test_images, test_labels = load_mnist_m_split(test_images_dir, test_labels_file)

    n_total = len(full_train_images)
    n_val = int(val_size * n_total)
    n_train = n_total - n_val

    generator = torch.Generator()
    perm = torch.randperm(n_total, generator=generator)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_images = full_train_images[train_idx].clone()
    train_labels = full_train_labels[train_idx].clone()
    val_images = full_train_images[val_idx].clone()
    val_labels = full_train_labels[val_idx].clone()

    mean, std = compute_channel_stats_hwc_uint8(train_images)

    print(f"Train dataset size: {len(train_images)}")
    print(f"Validation dataset size: {len(val_images)}")
    print(f"Test dataset size: {len(test_images)}")
    print(f"MNIST-M mean: {mean}")
    print(f"MNIST-M std: {std}")

    save_splits(
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels,
        output_path,
        {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "val_size": val_size,
            "mean": mean,
            "std": std,
        },
    )

def prepare_images(images, target_size, mean, std):
    images = images.float()

    if images.max() > 1.0:
        images = images / 255.0

    # MNIST: [N,H,W] -> [N,1,H,W]
    if images.ndim == 3:
        images = images.unsqueeze(1)

    # SVHN and MNIST-M: [N,H,W,C] -> [N,C,H,W]
    if images.ndim == 4 and images.shape[1] not in (1, 3):
        images = images.permute(0, 3, 1, 2)

    # MNIST: [N,1,H,W] -> [N,3,H,W]
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    
    if images.shape[-2:] != target_size:
        images = F.interpolate(
            images,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    mean_t = torch.tensor(mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)

    # MNIST mean and std shape 1, repeat to 3 channels
    if mean_t.shape[1] == 1 and images.shape[1] == 3:
        mean_t = mean_t.repeat(1, 3, 1, 1)
        std_t = std_t.repeat(1, 3, 1, 1)
    return (images - mean_t) / std_t

def load_data(data, input_dir, target_size=(32, 32)):
    input_path = Path(input_dir)
    split_path = input_path / f"{data}.pt"
    checkpoint = torch.load(split_path, map_location="cpu")

    config_path = input_path / "config.pt"
    config = torch.load(config_path, map_location="cpu")
    mean = config.get("mean")
    std = config.get("std")

    images = prepare_images(checkpoint["images"], target_size, mean, std)
    labels = checkpoint["labels"].long()

    return images, labels

# def load_loader(data, input_dir, target_size=(32, 32)):
#     input_path = Path(input_dir)
#     split_path = input_path / f"{data}.pt"
#     checkpoint = torch.load(split_path, map_location="cpu")

#     config_path = input_path / "config.pt"
#     config = torch.load(config_path, map_location="cpu")
#     batch_size = config.get("batch_size")
#     num_workers = config.get("num_workers")
#     mean = config.get("mean")
#     std = config.get("std")

#     ds = TensorDataset(prepare_images(checkpoint["images"], target_size, mean, std), 
#                        checkpoint["labels"].long())

#     return DataLoader(
#         ds,
#         batch_size=batch_size,
#         shuffle=(data == "train"),
#         num_workers=num_workers,
#         pin_memory=torch.cuda.is_available(),
#     )

def reconstruct_image_to_plot(img_tensor, mean, std):
    img = img_tensor.detach().cpu().float()

    mean_t = torch.tensor(mean, dtype=img.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype).view(-1, 1, 1)

    if mean_t.shape[0] == 1 and img.shape[0] == 3:
        mean_t = mean_t.repeat(3, 1, 1)
        std_t = std_t.repeat(3, 1, 1)

    img = img * std_t + mean_t
    img = img.clamp(0.0, 1.0)

    if img.shape[0] == 1:
        return img.squeeze(0).numpy(), "gray"

    return img.permute(1, 2, 0).numpy(), None

def plot_image(images, labels, input_dir, idx = 0):
    config = torch.load(Path(input_dir) / "config.pt", map_location="cpu")
    mean = config["mean"]
    std = config["std"]

    img, cmap = reconstruct_image_to_plot(images[idx], mean, std)
    label = labels[idx].item()

    plt.figure(figsize=(4, 4))
    if cmap is None:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)

    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()