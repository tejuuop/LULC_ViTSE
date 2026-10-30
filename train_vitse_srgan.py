# train_vitse.py
# ViT-SE + optional simplified Modified Firefly Algorithm for EuroSAT RGB

import os
import math
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.datasets as datasets

import json
import csv
from datetime import datetime

# -----------------------------
# Basic utilities
# -----------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_info(root):
    """
    Reads class folders.
    Returns samples list and class names.
    """
    full = datasets.ImageFolder(root)
    return full.samples, full.classes


def stratified_split(samples, train_ratio=0.8, seed=42):
    """
    80% train, 20% test.
    Keeps class balance.
    """
    targets = np.array([label for _, label in samples])
    rng = np.random.RandomState(seed)

    train_idx = []
    test_idx = []

    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)

        cut = int(len(idx) * train_ratio)
        train_idx.extend(idx[:cut].tolist())
        test_idx.extend(idx[cut:].tolist())

    return train_idx, test_idx


class FolderSubset(Dataset):
    """
    Subset of ImageFolder with transform.
    """
    def __init__(self, samples, indices, transform):
        self.samples = [samples[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def make_transforms(image_size):
    """
    Paper mentions flip, rotation, contrast changes.
    We use simple augmentations.
    """
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        normalize
    ])

    test_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, test_transform


# -----------------------------
# Model configuration
# -----------------------------

def get_config(name):
    """
    smoke = very fast test
    overnight = best realistic run for 8GB GPU
    paper384_small_model = keeps 384 input but smaller model, slow
    exact_table4 = paper Table 4, likely too heavy for 8GB
    """

    if name == "smoke":
        return {
            "image_size": 224,
            "patch_size": 16,
            "dim": 256,
            "depth": 6,
            "heads": 4,
            "mlp_dim": 512,
            "dropout": 0.2,
            "batch_size": 32,
            "epochs": 3,
            "lr": 0.0001
        }

    if name == "overnight":
        return {
            "image_size": 224,
            "patch_size": 16,
            "dim": 768,
            "depth": 12,
            "heads": 12,
            "mlp_dim": 3072,
            "dropout": 0.2,
            "batch_size": 16,
            "epochs": 100,
            "lr": 0.0001
        }

    if name == "paper384_small_model":
        return {
            "image_size": 384,
            "patch_size": 16,
            "dim": 512,
            "depth": 12,
            "heads": 8,
            "mlp_dim": 2048,
            "dropout": 0.2,
            "batch_size": 1,
            "epochs": 100,
            "lr": 0.0001
        }

    if name == "exact_table4":
        return {
            "image_size": 384,
            "patch_size": 16,
            "dim": 1024,
            "depth": 24,
            "heads": 16,
            "mlp_dim": 4096,
            "dropout": 0.2,
            "batch_size": 1,
            "epochs": 100,
            "lr": 0.0001
        }

    raise ValueError(f"Unknown config: {name}")

# -----------------------------
# ViT-SE model
# -----------------------------

class PatchEmbedding(nn.Module):
    """
    Cuts image into patches.
    Paper patch size = 16x16.
    """
    def __init__(self, image_size, patch_size, in_channels, dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: B, 3, H, W
        x = self.proj(x)          # B, dim, H/P, W/P
        x = x.flatten(2)          # B, dim, N
        x = x.transpose(1, 2)     # B, N, dim
        return x


class SinusoidalPositionEncoding(nn.Module):
    """
    Paper uses positional encoding.
    """
    def __init__(self, num_tokens, dim):
        super().__init__()

        pe = torch.zeros(num_tokens, dim)
        position = torch.arange(0, num_tokens).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    Simple meaning:
    It learns which feature channels are important.
    Then it increases important features and reduces useless ones.
    """
    def __init__(self, dim, reduction=16):
        super().__init__()

        hidden = max(8, dim // reduction)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: B, N, D

        # Squeeze: global average over tokens
        s = x.mean(dim=1)          # B, D

        # Excitation: channel weights
        w = self.net(s)            # B, D
        w = w.unsqueeze(1)         # B, 1, D

        # Scale features
        return x * w


class ViTSEBlock(nn.Module):
    """
    One Transformer encoder block + SE block.
    """
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()

        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.se = SEBlock(dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.se(x)
        return x


class SRResBlock(nn.Module):
    """
    Inspired by MF-SRGAN (Basu et al., 2026).
    Recovers high-frequency textures lost when resizing 64x64 EuroSAT images.
    Uses 1D convolutions to operate directly on ViT tokens.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        # SRGAN relies heavily on LeakyReLU for texture generation
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = x.transpose(1, 2)  # Conv1d expects (Batch, Dim, Tokens)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = x.transpose(1, 2)  # Back to (Batch, Tokens, Dim)
        x = self.norm2(x + res) # Residual connection
        return x


class ViTSE(nn.Module):
    """
    Modified Vision Transformer with SE blocks.
    """
    def __init__(self, cfg, num_classes):
        super().__init__()

        image_size = cfg["image_size"]
        patch_size = cfg["patch_size"]
        dim = cfg["dim"]
        depth = cfg["depth"]
        heads = cfg["heads"]
        mlp_dim = cfg["mlp_dim"]
        dropout = cfg["dropout"]

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            dim=dim
        )

        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_encode = SinusoidalPositionEncoding(
            num_tokens=num_patches + 1,
            dim=dim
        )

        self.dropout = nn.Dropout(dropout)

        # SRGAN-inspired Texture Refinement
        self.sr_refine = nn.Sequential(
            SRResBlock(dim),
            SRResBlock(dim)
        )

        self.blocks = nn.ModuleList([
            ViTSEBlock(
                dim=dim,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        b = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_encode(x)
        x = self.dropout(x)

        x = self.pos_encode(x)
        x = self.dropout(x)

        # Apply Super-Resolution Texture Refinement before Transformer
        x = self.sr_refine(x)

        for blk in self.blocks:
            x = blk(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # CLS token is used for classification
        cls_output = x[:, 0]

        return self.head(cls_output)


# -----------------------------
# Metrics
# -----------------------------

def metrics_from_confusion(conf):
    """
    Computes accuracy, precision, recall, F1, kappa.
    """
    eps = 1e-12

    tp = np.diag(conf).astype(np.float64)

    true = conf.sum(axis=1).astype(np.float64)
    pred = conf.sum(axis=0).astype(np.float64)

    fp = pred - tp
    fn = true - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    total = conf.sum()
    acc = tp.sum() / max(total, 1)

    pe = (true * pred).sum() / (max(total, 1) ** 2)
    kappa = (acc - pe) / (1 - pe + eps)

    return {
        "accuracy": float(acc),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "kappa": float(kappa)
    }



def create_run_dir(output_root):
    os.makedirs(output_root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, stamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_csv(path, row, fieldnames):
    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_split_csv(samples, train_idx, test_idx, classes, run_dir):
    train_path = os.path.join(run_dir, "train_split.csv")
    test_path = os.path.join(run_dir, "test_split.csv")

    with open(train_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "class"])

        for i in train_idx:
            path, label = samples[i]
            writer.writerow([path, classes[label]])

    with open(test_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "class"])

        for i in test_idx:
            path, label = samples[i]
            writer.writerow([path, classes[label]])


def save_confusion_csv(conf, classes, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class"] + list(classes))

        for i, cls in enumerate(classes):
            writer.writerow([cls] + conf[i].tolist())


def save_per_class_csv(conf, classes, path):
    eps = 1e-12

    tp = np.diag(conf).astype(np.float64)
    true = conf.sum(axis=1).astype(np.float64)
    pred = conf.sum(axis=0).astype(np.float64)

    fp = pred - tp
    fn = true - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1"])
        writer.writeheader()

        for i, cls in enumerate(classes):
            writer.writerow({
                "class": cls,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i])
            })


def evaluate_with_confusion(model, loader, device, num_classes):
    model.eval()

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()

            for t, p in zip(labels, preds):
                conf[t, p] += 1

    metrics = metrics_from_confusion(conf)

    return conf, metrics


def save_test_predictions(model, dataset, classes, device, path, batch_size=16):
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    rows = []
    idx = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()

            for true_label, pred_label in zip(labels, preds):
                file_path = dataset.samples[idx][0] if idx < len(dataset.samples) else ""

                rows.append({
                    "file": os.path.basename(file_path),
                    "true_class": classes[true_label],
                    "predicted_class": classes[pred_label],
                    "correct": int(true_label == pred_label)
                })

                idx += 1

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "true_class", "predicted_class", "correct"]
        )

        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# Training and evaluation
# -----------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()

    total_loss = 0.0
    correct = 0
    count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        count += images.size(0)

    avg_loss = total_loss / max(count, 1)
    avg_acc = correct / max(count, 1)

    return avg_loss, avg_acc


def evaluate(model, loader, device, num_classes):
    model.eval()

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()

            for t, p in zip(labels, preds):
                conf[t, p] += 1

    return metrics_from_confusion(conf)


# -----------------------------
# Optional simplified MFA
# -----------------------------

def evaluate_lr_fast(lr, args, cfg, samples, train_idx, test_idx, num_classes, device):
    """
    Fast evaluation for one learning rate.
    Used by MFA.
    """
    train_tf, test_tf = make_transforms(cfg["image_size"])

    train_set = FolderSubset(samples, train_idx, train_tf)
    test_set = FolderSubset(samples, test_idx, test_tf)

    train_take = min(args.mfa_train_samples, len(train_set))
    test_take = min(args.mfa_test_samples, len(test_set))

    train_set = Subset(train_set, list(range(train_take)))
    test_set = Subset(test_set, list(range(test_take)))

    batch_size = args.batch_size if args.batch_size else cfg["batch_size"]

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    model = ViTSE(cfg, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for _ in range(args.mfa_eval_epochs):
        train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler
        )

    metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=num_classes
    )

    return metrics["accuracy"]


def run_mfa(args, cfg, samples, train_idx, test_idx, num_classes, device):
    """
    Simplified Modified Firefly Algorithm.

    Paper exact MFA:
    - 50 fireflies
    - 700 iterations

    That is too slow for one sitting.
    Default here uses small numbers.
    """
    low = 1e-4
    high = 1e-2

    n = args.mfa_fireflies
    max_iter = args.mfa_iter

    gamma = 5.0
    beta0 = 0.4
    alpha = 0.01

    pos = np.random.uniform(low, high, size=(n, 1))

    best_pos = pos[0].copy()
    best_fit = -1.0

    for it in range(max_iter):
        fits = np.zeros(n)

        for i in range(n):
            lr = float(np.clip(pos[i, 0], low, high))
            print(f"MFA checking firefly {i + 1}/{n}, LR = {lr:.6f}")

            fits[i] = evaluate_lr_fast(
                lr=lr,
                args=args,
                cfg=cfg,
                samples=samples,
                train_idx=train_idx,
                test_idx=test_idx,
                num_classes=num_classes,
                device=device
            )

            if fits[i] > best_fit:
                best_fit = fits[i]
                best_pos = pos[i].copy()

        # Modified inertia factor
        w_max = 0.9
        w_min = 0.4
        w = w_max - (w_max - w_min) * it / (1 + max_iter)

        best_j = int(np.argmax(fits))

        for i in range(n):
            if best_j != i and fits[best_j] > fits[i]:
                d2 = float(np.sum((pos[i] - pos[best_j]) ** 2))

                attraction = beta0 * math.exp(-gamma * d2) * (pos[best_j] - pos[i])

                z = np.random.rand() * (it + 1) / max(1, max_iter)
                psi = (1 - math.exp(-z)) / (1 + math.exp(-z) + 1e-12)

                random_part = alpha * psi * (np.random.rand(1, 1) - 0.5)

                pos[i] = w * pos[i] + attraction + random_part
            else:
                random_part = alpha * (np.random.rand(1, 1) - 0.5)
                pos[i] = w * pos[i] + random_part

            pos[i] = np.clip(pos[i], low, high)

    return float(best_pos[0]), float(best_fit)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    # Use this if you run the script from inside working/
    parser.add_argument("--data", type=str, default="./EuroSAT-RGB")
    parser.add_argument("--output", type=str, default="./outputs")

    parser.add_argument("--config", type=str, default="smoke")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-predictions", dest="no_predictions", action="store_true")

    # Optional simplified MFA
    parser.add_argument("--mfa", action="store_true")
    parser.add_argument("--mfa-fireflies", dest="mfa_fireflies", type=int, default=3)
    parser.add_argument("--mfa-iter", dest="mfa_iter", type=int, default=2)
    parser.add_argument("--mfa-eval-epochs", dest="mfa_eval_epochs", type=int, default=1)
    parser.add_argument("--mfa-train-samples", dest="mfa_train_samples", type=int, default=768)
    parser.add_argument("--mfa-test-samples", dest="mfa_test_samples", type=int, default=384)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_dir = create_run_dir(args.output)

    print(f"Results will be saved in: {run_dir}")
    print(f"Device: {device}")

    cfg = get_config(args.config)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    if args.lr is not None and not args.mfa:
        cfg["lr"] = args.lr

    # Load dataset
    samples, classes = load_dataset_info(args.data)
    num_classes = len(classes)

    print(f"Found {len(samples)} images")
    print(f"Classes: {classes}")
    print(f"Number of classes: {num_classes}")

    # 80/20 split
    train_idx, test_idx = stratified_split(
        samples=samples,
        train_ratio=0.8,
        seed=args.seed
    )

    # Save train/test file lists
    save_split_csv(samples, train_idx, test_idx, classes, run_dir)

    # Optional MFA search
    if args.mfa:
        print("Starting simplified MFA learning-rate search...")

        best_lr, best_mfa_acc = run_mfa(
            args=args,
            cfg=cfg,
            samples=samples,
            train_idx=train_idx,
            test_idx=test_idx,
            num_classes=num_classes,
            device=device
        )

        cfg["lr"] = best_lr

        print(f"MFA best LR: {best_lr:.6f}")
        print(f"MFA quick accuracy: {best_mfa_acc:.4f}")

        save_json(
            os.path.join(run_dir, "mfa_result.json"),
            {
                "best_lr": best_lr,
                "quick_accuracy": best_mfa_acc
            }
        )

    # Data
    train_transform, test_transform = make_transforms(cfg["image_size"])

    train_set = FolderSubset(samples, train_idx, train_transform)
    test_set = FolderSubset(samples, test_idx, test_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    # Model
    model = ViTSE(cfg, num_classes).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.2f} M")

    # Save run config
    run_config = {
        "args": vars(args),
        "config": cfg,
        "classes": classes,
        "num_classes": num_classes,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "model_parameters_million": param_count
    }

    save_json(
        os.path.join(run_dir, "run_config.json"),
        run_config
    )

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    metrics_csv = os.path.join(run_dir, "metrics.csv")

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_acc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_kappa"
    ]

    best_acc = 0.0
    best_epoch = 0
    best_metrics = None

    final_metrics = None
    final_conf = None

    for epoch in range(cfg["epochs"]):
        train_loss, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler
        )

        conf, metrics = evaluate_with_confusion(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=num_classes
        )

        final_conf = conf
        final_metrics = metrics

        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_acc": float(metrics["accuracy"]),
            "test_precision": float(metrics["precision"]),
            "test_recall": float(metrics["recall"]),
            "test_f1": float(metrics["f1"]),
            "test_kappa": float(metrics["kappa"])
        }

        append_csv(metrics_csv, row, fieldnames)

        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(run_dir, "last_model.pth")
            )

        print(
            f"Epoch {epoch + 1}/{cfg['epochs']} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Test Acc {metrics['accuracy']:.4f} | "
            f"Test F1 {metrics['f1']:.4f} | "
            f"Kappa {metrics['kappa']:.4f}"
        )

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_epoch = epoch + 1
            best_metrics = metrics

            torch.save(
                model.state_dict(),
                os.path.join(run_dir, "best_model.pth")
            )

            save_confusion_csv(
                conf,
                classes,
                os.path.join(run_dir, "confusion_best.csv")
            )

            save_per_class_csv(
                conf,
                classes,
                os.path.join(run_dir, "per_class_best.csv")
            )

            best_save = {
                "epoch": best_epoch,
                "accuracy": best_acc
            }
            best_save.update(metrics)

            save_json(
                os.path.join(run_dir, "best_metrics.json"),
                best_save
            )

    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(run_dir, "last_model.pth")
    )

    # Save final results
    if final_conf is not None:
        save_confusion_csv(
            final_conf,
            classes,
            os.path.join(run_dir, "confusion_final.csv")
        )

        save_per_class_csv(
            final_conf,
            classes,
            os.path.join(run_dir, "per_class_final.csv")
        )

    if final_metrics is not None:
        save_json(
            os.path.join(run_dir, "final_metrics.json"),
            final_metrics
        )

    summary = {
        "results_folder": run_dir,
        "best_epoch": best_epoch,
        "best_accuracy": best_acc,
        "best_metrics": best_metrics
    }

    save_json(
        os.path.join(run_dir, "summary.json"),
        summary
    )

    # Save test predictions using best model
    if not args.no_predictions:
        best_path = os.path.join(run_dir, "best_model.pth")

        if os.path.isfile(best_path):
            model.load_state_dict(
                torch.load(best_path, map_location=device)
            )

        save_test_predictions(
            model=model,
            dataset=test_set,
            classes=classes,
            device=device,
            path=os.path.join(run_dir, "test_predictions.csv"),
            batch_size=min(cfg["batch_size"], 32)
        )

    print("Saved files:")

    for file_name in sorted(os.listdir(run_dir)):
        print(" -", file_name)

    print(f"Done. Results are inside: {run_dir}")


if __name__ == "__main__":
    main()
