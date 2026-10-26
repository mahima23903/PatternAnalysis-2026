from pathlib import Path


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