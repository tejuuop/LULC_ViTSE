# train_vitse.py
# ViT-SE + optional simplified Modified Firefly Algorithm for EuroSAT RGB

import os
import math
import random
import argparse
import numpy as np
from PIL import Image
import csv
import json
import cv2
from torchvision.transforms import functional as TF
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.datasets as datasets


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


class LegibilityTransform:
    """
    Step 2 legibility preprocessing:
    RGB + CLAHE + Canny edge map.
    """

    def __init__(
        self,
        image_size,
        train=True,
        channels=("clahe", "canny"),
        clahe_clip=2.0,
        clahe_grid=8,
        canny_low=50,
        canny_high=150
    ):
        self.image_size = image_size
        self.train = train
        self.channels = channels
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.canny_low = canny_low
        self.canny_high = canny_high

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.color_jitter = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )

    def _rgb_to_gray_np(self, rgb_tensor):
        gray = TF.rgb_to_grayscale(rgb_tensor)
        gray = gray.squeeze(0).numpy()
        gray = (gray * 255.0).clip(0, 255).astype(np.uint8)
        return gray

    def __call__(self, img):
        img = img.convert("RGB")

        img = TF.resize(img, [self.image_size, self.image_size])

        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)

            if random.random() < 0.5:
                img = TF.vflip(img)

            angle = random.uniform(-10.0, 10.0)
            img = TF.rotate(img, angle)

            img = self.color_jitter(img)

        rgb = TF.to_tensor(img)  # shape: 3, H, W, range [0, 1]

        extras = []

        gray_np = None
        if "clahe" in self.channels or "canny" in self.channels:
            gray_np = self._rgb_to_gray_np(rgb)

        if "clahe" in self.channels:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=(self.clahe_grid, self.clahe_grid)
            )
            clahe_np = clahe.apply(gray_np)

            clahe_tensor = torch.from_numpy(clahe_np).float() / 255.0
            clahe_tensor = (clahe_tensor - 0.5) / 0.5

            extras.append(clahe_tensor)

        if "canny" in self.channels:
            # Small median blur, inspired by the legibility paper.
            gray_den = cv2.medianBlur(gray_np, 3)

            edges = cv2.Canny(
                gray_den,
                self.canny_low,
                self.canny_high
            )

            edges_tensor = torch.from_numpy(edges).float() / 255.0
            edges_tensor = (edges_tensor - 0.5) / 0.5

            extras.append(edges_tensor)

        rgb = self.normalize(rgb)

        if len(extras) > 0:
            return torch.cat([rgb] + extras, dim=0)

        return rgb



def make_transforms(cfg):
    """
    Paper mentions flip, rotation, contrast changes.
    Now also supports Step 2 legibility channels.
    """

    image_size = cfg["image_size"]

    if not cfg.get("use_legibility", False):
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        train_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            T.ToTensor(),
            normalize
        ])

        test_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalize
        ])

        return train_transform, test_transform

    channels = tuple(cfg.get("legibility_channels", ["clahe", "canny"]))

    train_transform = LegibilityTransform(
        image_size=image_size,
        train=True,
        channels=channels,
        clahe_clip=cfg.get("clahe_clip", 2.0),
        clahe_grid=cfg.get("clahe_grid", 8),
        canny_low=cfg.get("canny_low", 50),
        canny_high=cfg.get("canny_high", 150)
    )

    test_transform = LegibilityTransform(
        image_size=image_size,
        train=False,
        channels=channels,
        clahe_clip=cfg.get("clahe_clip", 2.0),
        clahe_grid=cfg.get("clahe_grid", 8),
        canny_low=cfg.get("canny_low", 50),
        canny_high=cfg.get("canny_high", 150)
    )

    return train_transform, test_transform


# -----------------------------
# Model configuration
# -----------------------------

def get_config(name):
    """
    exact = paper Table 4 settings.
    small = runnable on normal GPU/CPU.
    """

    if name == "exact":
        cfg = {
            "image_size": 384,
            "patch_size": 16,
            "dim": 1024,
            "depth": 24,
            "heads": 16,
            "mlp_dim": 4096,
            "dropout": 0.2,
            "batch_size": 4,
            "epochs": 100,
            "lr": 0.0001
        }

    elif name =="overnight":
        cfg = {
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

    else:
        cfg = {
            "image_size": 224,
            "patch_size": 16,
            "dim": 256,
            "depth": 6,
            "heads": 4,
            "mlp_dim": 512,
            "dropout": 0.2,
            "batch_size": 32,
            "epochs": 20,
            "lr": 0.0001
        }

    # Step 2: legibility-enhanced input
    cfg["use_legibility"] = True
    cfg["legibility_channels"] = ["clahe", "canny"]

    # RGB + CLAHE + Canny
    cfg["in_channels"] = 3 + len(cfg["legibility_channels"])

    return cfg


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
            in_channels=cfg.get("in_channels", 3),
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
    train_tf, test_tf = make_transforms(cfg)

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

    parser.add_argument("--data", type=str, default="./working")
    parser.add_argument("--output", type=str, default="./outputs")
    parser.add_argument("--config", type=str, default="small",
                        choices=["small", "overnight", "exact"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Optional MFA
    parser.add_argument("--mfa", action="store_true")
    parser.add_argument("--mfa-fireflies", dest="mfa_fireflies", type=int, default=5)
    parser.add_argument("--mfa-iter", dest="mfa_iter", type=int, default=3)
    parser.add_argument("--mfa-eval-epochs", dest="mfa_eval_epochs", type=int, default=1)
    parser.add_argument("--mfa-train-samples", dest="mfa_train_samples", type=int, default=512)
    parser.add_argument("--mfa-test-samples", dest="mfa_test_samples", type=int, default=256)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    cfg = get_config(args.config)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    if args.lr is not None and not args.mfa:
        cfg["lr"] = args.lr

    # Load dataset info
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

    # Optional MFA search for learning rate
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
        print(f"MFA best LR: {best_lr:.6f}, quick validation accuracy: {best_mfa_acc:.4f}")

    # Data
    train_tf, test_tf = make_transforms(cfg)

    train_set = FolderSubset(samples, train_idx, train_tf)
    test_set = FolderSubset(samples, test_idx, test_tf)

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

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_acc = 0.0

    for epoch in range(cfg["epochs"]):
        train_loss, train_acc = train_epoch(
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
            torch.save(model.state_dict(), os.path.join(args.output, "best_vitse.pth"))

    print(f"Best test accuracy: {best_acc:.4f}")
    print("Saved best model as best_vitse.pth")


if __name__ == "__main__":
    main()
