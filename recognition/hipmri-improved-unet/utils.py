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