# HipMRI Segmentation Using an Improved U-Net

## 1. Introduction

Medical image segmentation involves identifying and classifying anatomical regions within medical images. Accurate segmentation is important for medical image analysis, as errors in identifying anatomical structures and their boundaries can affect the reliability of subsequent analysis.

U-Net is a widely used convolutional neural network architecture for medical image segmentation. Although conventional 2D U-Net models can achieve high segmentation accuracy, their predictions may still contain boundary inaccuracies, missed structures and false-positive regions. Furthermore, evaluating segmentation performance using only overall metrics such as Dice similarity can conceal errors affecting individual anatomical classes.

This project investigates an **Improved 2D U-Net architecture** for multi-class segmentation of HipMRI images. A standard 2D U-Net is implemented as the baseline, while the improved architecture incorporates residual context modules, learnable downsampling, localisation modules and multi-scale prediction fusion.

The primary objective is to investigate whether the Improved U-Net provides better segmentation performance and boundary precision than the standard U-Net while maintaining reasonable computational efficiency. Both architectures are evaluated using the same held-out test dataset, with comparisons based on Dice similarity, Intersection over Union (IoU), boundary precision, model complexity, GPU memory consumption and inference time.

In addition to quantitative evaluation, the project examines individual segmentation failures to identify limitations that may not be apparent from aggregate performance metrics.



# Feasibility Review — HipMRI Improved U-Net

**Project:** Prostate Radiotherapy Planning | **Difficulty:** Normal | **Model:** 2D Improved U-Net

### 1. User Need, Scope and Acceptance Criteria

The prototype assists imaging specialists by generating prostate radiotherapy contours for expert review. The engineering dilemma is whether Improved U-Net provides clinically meaningful boundary improvements over standard 2D U-Net at a reasonable computational cost.

**Acceptance Criteria:**
- Prostate-specific test Dice ≥0.75; additional foreground macro-Dice target ≥0.90.
- Compare both models using per-class Dice, IoU and boundary errors on identical patient-level splits.
- Analyse 3–5 difficult cases, particularly prostate base/apex and organ-boundary leakage.
- Measure peak GPU VRAM and per-case inference time.

### 2. Model Choice and Course Concepts

HipMRI provides 11,460 training, 660 validation and 540 test slices. The standard 2D U-Net is the baseline; the selected Improved U-Net uses residual context modules, instance normalization, skip connections and deep supervision, informed by Isensee et al. (2018).

Relevant concepts include CNNs, encoder–decoder segmentation, residual learning and regularization. The 2D approach is computationally efficient but lacks explicit inter-slice continuity.

### 3. Preliminary Feasibility Evidence

Data inspection identified inconsistent image dimensions (256 × 128 and 256 × 144), resolved using symmetric zero-padding for images and masks. Patient-level split separation was checked.

Both models have been trained. Improved U-Net completed 20 epochs on Rangpur in **27m 38s**. Its minimum validation loss was **0.0671 (epoch 4)**, with mean foreground validation Dice **0.8795**. Prostate-specific test performance remains unverified.

### 4. Risks, Budget, Fallback and Future Work

**Risks:** Class imbalance, boundary leakage, base/apex errors, overfitting and missing 3D context.

**Budget:** Approximately 28 minutes per 20-epoch Improved U-Net run; additional experiments depend on GPU availability.

**Fallback:** Retain standard U-Net if Improved U-Net offers insufficient accuracy or boundary benefits relative to computational cost.

**Planned Experiments:**
- **2D comparison:** Evaluate both models on the same held-out test set using prostate Dice, per-class Dice and IoU.
- **Loss functions:** Compare Cross-Entropy against Cross-Entropy + Dice Loss; optionally test weighted Cross-Entropy.
- **Boundary analysis:** Inspect 3–5 failures using ground-truth, prediction and error overlays.
- **Efficiency:** Compare peak GPU VRAM, training runtime and per-case inference time.
- **Optional 3D extension:** If time permits, test a lightweight 3D U-Net to assess inter-slice context and computational trade-offs.
- **Conclusion:** Recommend the more practical model based on segmentation quality, boundary reliability and resource cost.