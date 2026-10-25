import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

#We are going to compare in case of base model- 2D U-Net
from dataset import HipMRIDataset
from modules import ImprovedUNet
from baseline_unet.modules import UNet


def visualize_segmentation(image, ground_truth, baseline_prediction,
                           improved_prediction, filename):

    image = image.squeeze().cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    baseline_prediction = baseline_prediction.cpu().numpy()
    improved_prediction = improved_prediction.cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original MRI")

    axes[1].imshow(ground_truth, cmap="tab10", vmin=0, vmax=5)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(baseline_prediction, cmap="tab10", vmin=0, vmax=5)
    axes[2].set_title("Baseline U-Net")

    axes[3].imshow(improved_prediction, cmap="tab10", vmin=0, vmax=5)
    axes[3].set_title("Improved U-Net")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(filename)
    plt.tight_layout()

    output_dir = Path("failure_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"comparison_{Path(filename).name}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison: {output_path}")


#Loading specific MRI slice and generating predictions
def load_mri_slice(dataset, filename):
    for index, image_path in enumerate(dataset.image_files):
        if image_path.name == filename:
            image, mask = dataset[index]
            return image, mask

    raise FileNotFoundError(f"MRI slice not found: {filename}")

#for base model and improved model
def load_trained_model(model_class, checkpoint_path, device):
    model = model_class(in_channels=1, num_classes=6)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model

def predict_segmentation(model, image, device):
    # Add a batch dimension: [1, 256, 144]
    # becomes [1, 1, 256, 144]
    input_tensor = image.unsqueeze(0).to(device)

    # Disable gradient calculations during inference
    with torch.no_grad():
        output = model(input_tensor)

        # Select the class with the highest score for each pixel
        prediction = output.argmax(dim=1)

    # Remove batch dimension and return to CPU
    return prediction.squeeze(0).cpu()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = Path(
        "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    )

    dataset = HipMRIDataset(
        image_dir=dataset_root / "keras_slices_test",
        mask_dir=dataset_root / "keras_slices_seg_test"
    )

    # Load both trained models
    baseline_model = load_trained_model(
        UNet,
        "checkpoints_unet_20/best_unet.pth",
        device
    )

    improved_model = load_trained_model(
        ImprovedUNet,
        "checkpoints_improved_20/best_improved_unet.pth",
        device
    )

    filenames = [
        "case_040_week_0_slice_38.nii.gz",
        "case_040_week_1_slice_20.nii.gz",
        "case_040_week_2_slice_51.nii.gz"
    ]

    for filename in filenames:
        image, ground_truth = load_mri_slice(dataset, filename)

        baseline_prediction = predict_segmentation(
            baseline_model, image, device
        )

        improved_prediction = predict_segmentation(
            improved_model, image, device
        )

        visualize_segmentation(
            image,
            ground_truth,
            baseline_prediction,
            improved_prediction,
            filename
        )


if __name__ == "__main__":
    main()
