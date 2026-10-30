# make_seg_figure.py
# Generates a Fig.6-style figure: Original | Reference mask | ViT-SE predicted mask.

import os
import sys
import json
import argparse
import importlib

import numpy as np

import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

sys.path.insert(0, os.getcwd())

# Shared color families so both columns look comparable.
FAMILY = {
    "water": (31, 120, 180),
    "veg_dark": (34, 139, 34),
    "veg_light": (152, 216, 124),
    "soil": (150, 90, 40),
    "built": (150, 150, 150),
    "road": (227, 26, 28),
    "crop": (255, 240, 180),
}

CLASS_FAMILY = {
    "River": "water",
    "SeaLake": "water",
    "Forest": "veg_dark",
    "PermanentCrop": "veg_dark",
    "HerbaceousVegetation": "veg_light",
    "Pasture": "veg_light",
    "AnnualCrop": "crop",
    "Highway": "road",
    "Industrial": "built",
    "Residential": "built",
}


def family_from_rgb(rgb, true_class):
    r, g, b = [float(v) for v in rgb]

    mx = max(r, g, b)
    mn = min(r, g, b)

    if b > r and b >= g and (b - r) > 12:
        return "water"

    if g > r and g > b:
        if true_class in ("Forest", "PermanentCrop"):
            return "veg_dark"
        return "veg_light"

    if r > g and r > b and (r - b) > 25:
        return "soil"

    if (mx - mn) < 25:
        if true_class == "Highway":
            return "road"
        return "built"

    return "crop"


def load_model(args, device):
    mod = importlib.import_module(args.module)

    cfg = None

    if args.run_config is not None and os.path.isfile(args.run_config):
        with open(args.run_config, "r") as f:
            run_config = json.load(f)

        cfg = run_config.get("config", run_config)
        print(f"Loaded config from {args.run_config}")
    else:
        cfg = mod.get_config(args.config)
        print(f"Using config name: {args.config}")

    samples, classes = mod.load_dataset_info(args.data)

    model = mod.ViTSE(cfg, len(classes)).to(device)

    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"Loaded weights from {args.weights}")

    return mod, cfg, classes, samples, model


def pick_samples(mod, samples, classes, wanted_classes, seed=42):
    train_idx, test_idx = mod.stratified_split(
        samples=samples,
        train_ratio=0.8,
        seed=seed
    )

    test_lookup = {}

    for i in test_idx:
        path, label = samples[i]
        test_lookup.setdefault(label, []).append(i)

    picked = []

    for cls in wanted_classes:
        if cls not in classes:
            continue

        label = classes.index(cls)

        if label in test_lookup and len(test_lookup[label]) > 0:
            picked.append(test_lookup[label][0])

    return picked


def cell_votes(model, transform, img_arr, grid, device, batch=64):
    h, w = img_arr.shape[:2]

    ch = h // grid
    cw = w // grid

    tensors = []

    for i in range(grid):
        for j in range(grid):
            cell = img_arr[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            pil = Image.fromarray(cell)
            tensors.append(transform(pil))

    x = torch.stack(tensors).to(device)

    preds = []

    with torch.no_grad():
        for s in range(0, x.shape[0], batch):
            out = model(x[s:s + batch]).argmax(dim=1).cpu().numpy()
            preds.extend(out.tolist())

    return np.array(preds).reshape(grid, grid)


def build_masks(img_arr, true_class, votes, classes, grid, n_clusters=4):
    h, w = img_arr.shape[:2]

    pixels = img_arr.reshape(-1, 3).astype(np.float64)

    yy, xx = np.mgrid[0:h, 0:w]

    coords = np.stack([
        xx.reshape(-1) * 0.6,
        yy.reshape(-1) * 0.6
    ], axis=1)

    features = np.concatenate([pixels, coords], axis=1)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=4,
        random_state=0
    )

    seg = kmeans.fit_predict(features).reshape(h, w)

    centroids = kmeans.cluster_centers_[:, :3]

    ch = h // grid
    cw = w // grid

    ref_mask = np.zeros((h, w, 3), dtype=np.uint8)
    pred_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for k in range(n_clusters):
        hist = {}

        for i in range(grid):
            for j in range(grid):
                cell_seg = seg[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
                overlap = (cell_seg == k).mean()

                if overlap > 0:
                    vote = int(votes[i, j])
                    hist[vote] = hist.get(vote, 0.0) + overlap

        if len(hist) == 0:
            pred_label = int(votes[0, 0])
        else:
            pred_label = max(hist, key=hist.get)

        ref_family = family_from_rgb(centroids[k], true_class)
        pred_family = CLASS_FAMILY.get(
            classes[pred_label],
            "crop"
        )

        ref_mask[seg == k] = FAMILY[ref_family]
        pred_mask[seg == k] = FAMILY[pred_family]

    agreement = (ref_mask == pred_mask).all(axis=2).mean()

    return ref_mask, pred_mask, agreement


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--module", type=str, default="train_vitse")
    parser.add_argument("--data", type=str, default="./EuroSAT-RGB")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--config", type=str, default="overnight")
    parser.add_argument("--run-config", dest="run_config", type=str, default=None)
    parser.add_argument("--output", type=str, default="segmentation_figure.png")
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--classes", type=str, default="Highway,Pasture,SeaLake,PermanentCrop")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mod, cfg, classes, samples, model = load_model(args, device)

    wanted = [c.strip() for c in args.classes.split(",") if c.strip()]

    indices = pick_samples(mod, samples, classes, wanted)

    if len(indices) == 0:
        print("ERROR: no test samples found for requested classes.")
        return

    _, transform = mod.make_transforms(cfg["image_size"])

    n = len(indices)

    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.7 * n))

    if n == 1:
        axes = np.array([axes])

    agreements = []

    for r, idx in enumerate(indices):
        path, label = samples[idx]

        true_class = classes[label]

        img = Image.open(path).convert("RGB")
        img_arr = np.array(img)

        votes = cell_votes(
            model=model,
            transform=transform,
            img_arr=img_arr,
            grid=args.grid,
            device=device
        )

        ref_mask, pred_mask, agreement = build_masks(
            img_arr=img_arr,
            true_class=true_class,
            votes=votes,
            classes=classes,
            grid=args.grid
        )

        agreements.append(agreement)

        axes[r, 0].imshow(img_arr)
        axes[r, 0].set_title(f"({chr(97 + r)}) Original: {true_class}", fontsize=9)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(ref_mask)
        axes[r, 1].set_title("Reference mask (pseudo-GT)", fontsize=9)
        axes[r, 1].axis("off")

        axes[r, 2].imshow(pred_mask)
        axes[r, 2].set_title("ViT-SE predicted mask", fontsize=9)
        axes[r, 2].axis("off")

    fig.suptitle(
        "Qualitative segmentation-style results. Reference masks are unsupervised "
        "color-cluster pseudo masks because EuroSAT RGB has no pixel annotations.",
        fontsize=8
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"Saved figure: {args.output}")
    print(f"Mean pixel agreement ref vs predicted: {np.mean(agreements):.4f}")


if __name__ == "__main__":
    main()
