from pathlib import Path

import nibabel as nib
import numpy as np


# Verify patient-level separation to prevent data leakage
# between training, validation, and test datasets.
def extract_patient_id(filename):
    """
    Extract patient ID from a HipMRI filename.

    Example:
        case_040_week_0_slice_38.nii.gz -> case_040
    """
    parts = Path(filename).name.split("_")

    if len(parts) < 6 or parts[0] != "case":
        raise ValueError(f"Unexpected filename format: {filename}")

    return f"{parts[0]}_{parts[1]}"


def verify_patient_split(train_dir, val_dir, test_dir):
    """
    Verify that patients do not overlap between dataset splits.
    """

    def get_patient_ids(directory):
        files = Path(directory).glob("case_*.nii.gz")
        return {extract_patient_id(file.name) for file in files}

    train_patients = get_patient_ids(train_dir)
    val_patients = get_patient_ids(val_dir)
    test_patients = get_patient_ids(test_dir)

    overlaps = {
        "Train-Val": train_patients & val_patients,
        "Train-Test": train_patients & test_patients,
        "Val-Test": val_patients & test_patients,
    }

    print("Patient-level split verification")
    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")

    for split, patients in overlaps.items():
        print(f"{split} overlap: {len(patients)}")

    return overlaps


def verify_image_mask_pairs(image_dir, mask_dir):
    """
    Verify that every MRI image has a corresponding segmentation mask
    and that no unmatched masks exist.

    Expected naming:
        Image: case_040_week_0_slice_38.nii.gz
        Mask:  seg_040_week_0_slice_38.nii.gz
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_files = list(image_dir.glob("case_*.nii.gz"))
    mask_files = list(mask_dir.glob("seg_*.nii.gz"))

    image_keys = {
        path.name.replace("case_", "", 1)
        for path in image_files
    }

    mask_keys = {
        path.name.replace("seg_", "", 1)
        for path in mask_files
    }

    missing_masks = image_keys - mask_keys
    missing_images = mask_keys - image_keys

    print("\nImage-mask pairing verification")
    print(f"MRI images: {len(image_files)}")
    print(f"Segmentation masks: {len(mask_files)}")
    print(f"Missing masks: {len(missing_masks)}")
    print(f"Missing images: {len(missing_images)}")

    return {
        "missing_masks": missing_masks,
        "missing_images": missing_images,
    }

def verify_spatial_alignment(image_dir, mask_dir):
    """
    Verify spatial consistency between MRI images and segmentation masks.

    Checks:
        1. Image and mask dimensions
        2. Affine transformation matrices
        3. Anatomical orientation
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    shape_mismatches = []
    affine_mismatches = []
    orientation_mismatches = []

    image_files = sorted(image_dir.glob("case_*.nii.gz"))

    for image_path in image_files:
        mask_name = image_path.name.replace("case_", "seg_", 1)
        mask_path = mask_dir / mask_name

        if not mask_path.exists():
            continue

        image = nib.load(str(image_path))
        mask = nib.load(str(mask_path))

        # Check whether image and mask have identical dimensions.
        if image.shape != mask.shape:
            shape_mismatches.append(image_path.name)

        # Check whether image and mask use matching spatial transforms.
        if not np.allclose(image.affine, mask.affine):
            affine_mismatches.append(image_path.name)

        # Check whether image and mask have matching orientations.
        image_orientation = nib.aff2axcodes(image.affine)
        mask_orientation = nib.aff2axcodes(mask.affine)

        if image_orientation != mask_orientation:
            orientation_mismatches.append(image_path.name)

    print("\nSpatial alignment verification")
    print(f"Image-mask pairs checked: {len(image_files)}")
    print(f"Shape mismatches: {len(shape_mismatches)}")
    print(f"Affine mismatches: {len(affine_mismatches)}")
    print(f"Orientation mismatches: {len(orientation_mismatches)}")

    return {
        "shape_mismatches": shape_mismatches,
        "affine_mismatches": affine_mismatches,
        "orientation_mismatches": orientation_mismatches,
    }