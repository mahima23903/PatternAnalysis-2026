from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class HipMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_files = sorted(
            self.image_dir.glob("case_*.nii.gz")
        )

        if not self.image_files:
            raise ValueError("No MRI images found")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]

        # Find the corresponding segmentation mask
        mask_name = image_path.name.replace("case_", "seg_", 1)
        mask_path = self.mask_dir / mask_name

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        # Load image and mask
        image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
        mask = np.asarray(
            nib.load(str(mask_path)).dataobj,
            dtype=np.int64
        )

        # Normalize image intensities
        minimum = image.min()
        maximum = image.max()

        if maximum > minimum:
            image = (image - minimum) / (maximum - minimum)
        else:
            image = np.zeros_like(image)

        # Convert arrays into PyTorch tensors
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        mask = torch.from_numpy(mask.copy()).long()

        return image, mask

