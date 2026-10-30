# run_best_model.py
# Loads a saved best model and runs evaluation or single-image prediction.

import argparse
import importlib
import json
import numpy as np

import torch
from PIL import Image
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()

    # Use the Python file that contains the model class.
    # For your SRGAN-modified model, use train_vitse_srgan.
    # For the old baseline model, use train_vitse.
    parser.add_argument("--module", type=str, default="train_vitse_srgan")

    parser.add_argument("--data", type=str, default="./EuroSAT-RGB")
    parser.add_argument("--weights", type=str, required=True)

    # Config name is used only if --run-config is not given.
    parser.add_argument("--config", type=str, default="overnight")

    # Best option: use the run_config.json saved inside outputs/<run_folder>.
    parser.add_argument("--run-config", dest="run_config", type=str, default=None)

    # If given, predict only this image.
    parser.add_argument("--image", type=str, default=None)

    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="best_model_predictions.csv")

    args = parser.parse_args()

    # Import model code
    mod = importlib.import_module(args.module)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    if args.run_config is not None:
        with open(args.run_config, "r") as f:
            run_config = json.load(f)

        cfg = run_config["config"]
        print(f"Loaded config from: {args.run_config}")
    else:
        cfg = mod.get_config(args.config)
        print(f"Using config name: {args.config}")

    # Load dataset info
    samples, classes = mod.load_dataset_info(args.data)
    num_classes = len(classes)

    print(f"Found dataset classes: {classes}")

    # Build model
    model = mod.ViTSE(cfg, num_classes).to(device)

    # Load saved weights
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"Loaded model weights from: {args.weights}")

    # -------------------------
    # Single image prediction
    # -------------------------
    if args.image is not None:
        _, test_transform = mod.make_transforms(cfg["image_size"])

        img = Image.open(args.image).convert("RGB")
        x = test_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(logits.argmax(dim=1).item())

        print(f"Image: {args.image}")
        print(f"Predicted class: {classes[pred_idx]}")
        print(f"Probability: {probs[pred_idx].item():.4f}")

        print("Top probabilities:")

        top = torch.topk(probs, k=min(5, num_classes))

        for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {classes[idx]}: {prob:.4f}")

        return

    # -------------------------
    # Full test evaluation
    # -------------------------
    train_idx, test_idx = mod.stratified_split(
        samples=samples,
        train_ratio=0.8,
        seed=42
    )

    _, test_transform = mod.make_transforms(cfg["image_size"])

    test_set = mod.FolderSubset(samples, test_idx, test_transform)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    rows = []
    idx = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()

            for true_label, pred_label in zip(labels_np, preds):
                conf[true_label, pred_label] += 1

                if idx < len(test_set.samples):
                    file_path = test_set.samples[idx][0]
                else:
                    file_path = ""

                rows.append({
                    "file": file_path,
                    "true_class": classes[true_label],
                    "predicted_class": classes[pred_label],
                    "correct": int(true_label == pred_label)
                })

                idx += 1

    metrics = mod.metrics_from_confusion(conf)

    print("Test metrics:")

    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save predictions
    with open(args.output, "w", newline="") as f:
        import csv

        writer = csv.DictWriter(
            f,
            fieldnames=["file", "true_class", "predicted_class", "correct"]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
