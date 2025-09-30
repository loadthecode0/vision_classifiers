import json
import csv
import os
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ensure_dir_exists(directory_path: str) -> None:
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def save_metrics_json(metrics: Dict[str, Any], output_path: str) -> None:
    ensure_dir_exists(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_metrics_csv(rows: List[Dict[str, Any]], output_path: str, fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir_exists(os.path.dirname(output_path))
    if not rows:
        # Write an empty file with headers if provided
        with open(output_path, 'w', newline='') as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        return
    if fieldnames is None:
        # Derive fieldnames from first row preserving insertion order
        fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    output_path: str,
    title: str = 'Training Curves',
    topk: Optional[int] = None,
    train_topk_accuracies: Optional[List[float]] = None,
    val_topk_accuracies: Optional[List[float]] = None,
) -> None:
    ensure_dir_exists(os.path.dirname(output_path))
    epochs_range = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(epochs_range, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs_range, val_losses, 'r-s', label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 4)

    # Accuracy (Top-1 and optional Top-k)
    ax2.plot(epochs_range, train_accuracies, 'b-o', label='Train Acc (Top-1)')
    ax2.plot(epochs_range, val_accuracies, 'r-s', label='Val Acc (Top-1)')
    if topk and topk > 1 and train_topk_accuracies and val_topk_accuracies:
        ax2.plot(epochs_range, train_topk_accuracies, 'c-^', label=f'Train Top-{topk}')
        ax2.plot(epochs_range, val_topk_accuracies, 'm-*', label=f'Val Top-{topk}')
        ax2.set_title(f'Top-1 and Top-{topk} Accuracy')
    else:
        ax2.set_title('Top-1 Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


