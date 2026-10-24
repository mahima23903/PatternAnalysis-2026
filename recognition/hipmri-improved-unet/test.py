import argparse #accept dataset and checkpoint paths when running scripts
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import ImprovedUNet
from train import validate

import numpy as np

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


#Calculate dice scores for an individual MRI slice
def calculate_slice_dice(prediction, target, class_id):
    pred_mask = prediction == class_id
    true_mask = target == class_id

    intersection = (pred_mask & true_mask).sum().item()
    denominator = pred_mask.sum().item() + true_mask.sum().item()

    if denominator == 0:
        return None

    return 2 * intersection / denominator

def find_failure_cases(model, loader, device, class_id=5):
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            for i in range(len(predictions)):
                score = calculate_slice_dice(
                    predictions[i],
                    masks[i],
                    class_id
                )

                if score is not None:
                    index = batch_idx * loader.batch_size + i
                    filename = loader.dataset.image_files[index].name

                    results.append((filename, score))

    results.sort(key=lambda item: item[1])

    print(f"\nLowest Dice scores for Class {class_id}:")

    for filename, score in results[:5]:
        print(f"{filename}: Dice = {score:.4f}")

    return results

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

    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_improved_20/best_improved_unet.pth"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Evaluating split: {args.split}")

    model = load_model(args.checkpoint, device)

    loader = load_data(
        args.data_root,
        split=args.split
    )

    evaluate_model(model, loader, device)
    if args.split == "test":
        find_failure_cases(
            model,
            loader,
            device,
            class_id=5
        )


if __name__ == "__main__":
    main()