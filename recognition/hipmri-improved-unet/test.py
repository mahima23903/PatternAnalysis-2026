import argparse #accept dataset and checkpoint paths when running scripts
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import ImprovedUNet
from train import validate

def load_model(checkpoint_path, device):
    # Initialise the same architecture used during training
    model = ImprovedUNet(in_channels=1, num_classes=6)

    # Load the saved model weights
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    # Move the model to the selected device
    model = model.to(device)

    # Switch to evaluation mode
    model.eval()

    return model

#dataloader to load the test and the validation data.
def load_data(data_root, split="validate", batch_size=16):
    root = Path(data_root)

    dataset = HipMRIDataset(
        root / f"keras_slices_{split}",
        root / f"keras_slices_seg_{split}"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return loader

# Evaluate the saved model using either validation or test data
def evaluate_model(model, loader, device):
    criterion = nn.CrossEntropyLoss()

    loss, mean_dice, mean_iou, dice_scores, iou_scores = validate(
        model,
        loader,
        criterion,
        device,
        num_classes=6
    )

    print(f"Loss: {loss:.4f}")
    print(f"Mean Foreground Dice: {mean_dice:.4f}")
    print(f"Mean Foreground IoU: {mean_iou:.4f}")

    for class_id in range(1, 6):
        dice = dice_scores[class_id]
        iou = iou_scores[class_id]

        if dice is not None and iou is not None:
            print(f"Class {class_id}: Dice = {dice:.4f}, IoU = {iou:.4f}")
        else:
            print(f"Class {class_id}: No valid pixels for evaluation")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Improved U-Net model"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["validate", "test"],
        default="validate"
    )

    args = parser.parse_args()