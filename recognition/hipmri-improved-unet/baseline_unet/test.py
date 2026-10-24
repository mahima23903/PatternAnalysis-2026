import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from baseline_unet.modules import UNet
from baseline_unet.train import validate

def load_model(checkpoint_path, device):
    # Initialise the baseline U-Net architecture
    model = UNet(in_channels=1, num_classes=6)

    # Load the saved checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    # Restore the trained model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move the model to the selected device
    model = model.to(device)

    # Switch to evaluation mode
    model.eval()

    return model