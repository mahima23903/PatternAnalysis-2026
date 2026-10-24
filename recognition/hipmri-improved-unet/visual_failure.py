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


#Loading specific MRI slice and generating predictions
def load_mri_slice(dataset, filename):
    for index, image_path in enumerate(dataset.image_files):
        if image_path.name == filename:
            image, mask = dataset[index]
            return image, mask

    raise FileNotFoundError(f"MRI slice not found: {filename}")

def load_trained_model(checkpoint_path, device):
    model = ImprovedUNet(in_channels=1, num_classes=6)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model

