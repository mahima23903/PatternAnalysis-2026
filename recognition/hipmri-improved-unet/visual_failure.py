import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from dataset import HipMRIDataset
from modules import ImprovedUNet


def visualize_segmentation(image, ground_truth, prediction, filename):
    # Convert tensors to NumPy arrays
    image = image.squeeze().cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    prediction = prediction.cpu().numpy()

    # Create three side-by-side plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original MRI")

    axes[1].imshow(ground_truth, cmap="tab10", vmin=0, vmax=5)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(prediction, cmap="tab10", vmin=0, vmax=5)
    axes[2].set_title("Improved U-Net Prediction")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(filename)
    plt.tight_layout()

    output_dir = Path("failure_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{Path(filename).name}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization: {output_path}")

