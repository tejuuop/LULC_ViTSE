# train_vitse_updated.py
# Optimized ViT-SE + MFA + Monte Carlo for EuroSAT on RTX 4060 (8GB VRAM)

import os
import math
import time
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


# -----------------------------
# Basic Utilities & Seed
# -----------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_info(root):
    full = datasets.ImageFolder(root)
    return full.samples, full.classes


def stratified_split(samples, train_ratio=0.8, seed=42):
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
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
# Model Configurations
# -----------------------------

def get_config(name):
    if name == "paper_4m":
        # Matches Table 11 parameter count (~4.3M params)
        return {
            "image_size": 224,
            "patch_size": 16,
            "dim": 256,
            "depth": 8,
            "heads": 8,
            "mlp_dim": 512,
            "dropout": 0.2,
            "batch_size": 64,
            "epochs": 100,
            "lr": 0.0001
        }
    elif name == "paper_table4":
        # Table 4 settings (~303M params) - WARNING: May OOM or run very slow
        return {
            "image_size": 384,
            "patch_size": 16,
            "dim": 1024,
            "depth": 24,
            "heads": 16,
            "mlp_dim": 4096,
            "dropout": 0.2,
            "batch_size": 16,
            "epochs": 100,
            "lr": 0.0001
        }
    else:
        # Fast testing configuration
        return {
            "image_size": 224,
            "patch_size": 16,
            "dim": 192,
            "depth": 6,
            "heads": 6,
            "mlp_dim": 384,
            "dropout": 0.2,
            "batch_size": 64,
            "epochs": 20,
            "lr": 0.0001
        }


# -----------------------------
# ViT-SE Architecture
# -----------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SinusoidalPositionEncoding(nn.Module):
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
        s = x.mean(dim=1)
        w = self.net(s).unsqueeze(1)
        return x * w


class ViTSEBlock(nn.Module):
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


class ViTSE(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        image_size = cfg["image_size"]
        patch_size = cfg["patch_size"]
        dim = cfg["dim"]
        depth = cfg["depth"]
        heads = cfg["heads"]
        mlp_dim = cfg["mlp_dim"]
        dropout = cfg["dropout"]

        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, dim)
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_encode = SinusoidalPositionEncoding(num_patches + 1, dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ViTSEBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
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

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])


# -----------------------------
# Metrics Calculation
# -----------------------------

def metrics_from_confusion(conf):
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


# -----------------------------
# Training & Evaluation Engine
# -----------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, count = 0.0, 0, 0

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

    return total_loss / max(count, 1), correct / max(count, 1)


def evaluate(model, loader, device, num_classes):
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            for t, p in zip(labels, preds):
                conf[t, p] += 1

    return metrics_from_confusion(conf)


# -----------------------------
# Modified Firefly Algorithm (MFA)
# -----------------------------

def run_mfa(args, cfg, samples, train_idx, test_idx, num_classes, device):
    low, high = 1e-4, 1e-2
    n = args.mfa_fireflies
    max_iter = args.mfa_iter
    gamma, beta0, alpha = 5.0, 0.4, 0.01

    pos = np.random.uniform(low, high, size=(n, 1))
    best_pos = pos[0].copy()
    best_fit = -1.0

    print(f"\n[MFA] Starting search with {n} fireflies over {max_iter} iterations...")

    train_tf, test_tf = make_transforms(cfg["image_size"])
    train_set = FolderSubset(samples, train_idx, train_tf)
    test_set = FolderSubset(samples, test_idx, test_tf)

    train_set = Subset(train_set, list(range(min(args.mfa_train_samples, len(train_set)))))
    test_set = Subset(test_set, list(range(min(args.mfa_test_samples, len(test_set)))))

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=args.workers)

    for it in range(max_iter):
        fits = np.zeros(n)
        for i in range(n):
            lr = float(np.clip(pos[i, 0], low, high))
            model = ViTSE(cfg, num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

            for _ in range(args.mfa_eval_epochs):
                train_epoch(model, train_loader, optimizer, criterion, device, scaler)

            metrics = evaluate(model, test_loader, device, num_classes)
            fits[i] = metrics["accuracy"]

            if fits[i] > best_fit:
                best_fit = fits[i]
                best_pos = pos[i].copy()

        w_max, w_min = 0.9, 0.4
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
                pos[i] = w * pos[i] + alpha * (np.random.rand(1, 1) - 0.5)
            pos[i] = np.clip(pos[i], low, high)

    return float(best_pos[0]), float(best_fit)


# -----------------------------
# Main Training Loop
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./working")
    parser.add_argument("--config", type=str, default="paper_4m", choices=["paper_4m", "paper_table4", "small"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # MFA Options
    parser.add_argument("--mfa", action="store_true")
    parser.add_argument("--mfa-fireflies", dest="mfa_fireflies", type=int, default=5)
    parser.add_argument("--mfa-iter", dest="mfa_iter", type=int, default=3)
    parser.add_argument("--mfa-eval-epochs", dest="mfa_eval_epochs", type=int, default=1)
    parser.add_argument("--mfa-train-samples", dest="mfa_train_samples", type=int, default=1024)
    parser.add_argument("--mfa-test-samples", dest="mfa_test_samples", type=int, default=512)

    # Monte Carlo Options
    parser.add_argument("--monte-carlo", dest="monte_carlo", type=int, default=1)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device)
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    cfg = get_config(args.config)
    if args.epochs is not None: cfg["epochs"] = args.epochs
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.lr is not None and not args.mfa: cfg["lr"] = args.lr

    samples, classes = load_dataset_info(args.data)
    num_classes = len(classes)
    print(f"Found {len(samples)} images across {num_classes} classes.")

    train_idx, test_idx = stratified_split(samples, train_ratio=0.8, seed=args.seed)

    if args.mfa:
        best_lr, best_mfa_acc = run_mfa(args, cfg, samples, train_idx, test_idx, num_classes, device)
        cfg["lr"] = best_lr
        print(f"[MFA Complete] Selected Learning Rate: {best_lr:.6f} | Val Acc: {best_mfa_acc:.4f}")

    train_tf, test_tf = make_transforms(cfg["image_size"])
    train_set = FolderSubset(samples, train_idx, train_tf)
    test_set = FolderSubset(samples, test_idx, test_tf)

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=args.workers, pin_memory=True)

    mc_accuracies, mc_f1s, mc_kappas = [], [], []

    print(f"\nStarting {args.monte_carlo} Run(s)... Config: {args.config} | Epochs per run: {cfg['epochs']}")

    for run in range(args.monte_carlo):
        set_seed(args.seed + run)
        model = ViTSE(cfg, num_classes).to(device)

        if run == 0:
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"Model Parameters: {params_m:.2f} Million")

        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        best_acc = 0.0
        start_time = time.time()

        for epoch in range(cfg["epochs"]):
            t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
            metrics = evaluate(model, test_loader, device, num_classes)

            if (epoch + 1) % 5 == 0 or epoch == cfg["epochs"] - 1:
                print(
                    f"Run {run + 1}/{args.monte_carlo} | Epoch {epoch + 1:03d}/{cfg['epochs']} | "
                    f"Train Loss {t_loss:.4f} | Train Acc {t_acc:.4f} | "
                    f"Test Acc {metrics['accuracy']:.4f} | Test F1 {metrics['f1']:.4f} | Kappa {metrics['kappa']:.4f}"
                )

            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                torch.save(model.state_dict(), "best_vitse.pth")

        elapsed = time.time() - start_time
        print(f"Run {run + 1} Best Test Accuracy: {best_acc:.4f} (Elapsed: {elapsed:.1f}s)")
        mc_accuracies.append(best_acc)
        mc_f1s.append(metrics["f1"])
        mc_kappas.append(metrics["kappa"])

    if args.monte_carlo > 1:
        print("\n=== Monte Carlo Stability Summary ===")
        print(f"Accuracy : Mean = {np.mean(mc_accuracies):.4f} | Std = {np.std(mc_accuracies):.4f} | Min = {np.min(mc_accuracies):.4f} | Max = {np.max(mc_accuracies):.4f}")
        print(f"F1-Score : Mean = {np.mean(mc_f1s):.4f} | Std = {np.std(mc_f1s):.4f}")
        print(f"Kappa    : Mean = {np.mean(mc_kappas):.4f} | Std = {np.std(mc_kappas):.4f}")

    print("\nSaved best model weights to best_vitse.pth")


if __name__ == "__main__":
    main()
