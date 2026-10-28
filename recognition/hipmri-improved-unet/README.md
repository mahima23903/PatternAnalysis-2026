# HipMRI Segmentation Using an Improved U-Net

## 1. Introduction

Medical image segmentation involves identifying and classifying anatomical regions within medical images. Accurate segmentation is important for medical image analysis, as errors in identifying anatomical structures and their boundaries can affect the reliability of subsequent analysis.

U-Net is a widely used convolutional neural network architecture for medical image segmentation. Although conventional 2D U-Net models can achieve high segmentation accuracy, their predictions may still contain boundary inaccuracies, missed structures and false-positive regions. Furthermore, evaluating segmentation performance using only overall metrics such as Dice similarity can conceal errors affecting individual anatomical classes.

This project investigates an **Improved 2D U-Net architecture** for multi-class segmentation of HipMRI images. A standard 2D U-Net is implemented as the baseline, while the improved architecture incorporates residual context modules, learnable downsampling, localisation modules and multi-scale prediction fusion.

The primary objective is to investigate whether the Improved U-Net provides better segmentation performance and boundary precision than the standard U-Net while maintaining reasonable computational efficiency. Both architectures are evaluated using the same held-out test dataset, with comparisons based on Dice similarity, Intersection over Union (IoU), boundary precision, model complexity, GPU memory consumption and inference time.

In addition to quantitative evaluation, the project examines individual segmentation failures to identify limitations that may not be apparent from aggregate performance metrics. Based on the experimental findings, the project assesses whether the Improved U-Net offers a suitable balance between segmentation accuracy, boundary reliability and computational efficiency, while identifying the limitations and improvements that must be addressed before practical deployment.

## 2. Project Structure

The project is organised into separate modules for dataset preparation, model implementation, training, evaluation and failure-case analysis. The baseline U-Net implementation is maintained in a separate directory to support comparisons with the Improved U-Net.

```text
hipmri-improved-unet/
│
├── README.md
├── dataset.py
├── modules.py
├── train.py
├── test.py
├── utils.py
├── visual_failure.py
│
└── baseline_unet/
    ├── modules.py
    ├── train.py
    └── test.py
```

**File descriptions:**

- `README.md` — Project documentation, experimental results and research findings.
- `dataset.py` — Loads HipMRI images and segmentation masks and applies preprocessing.
- `modules.py` — Defines the Improved U-Net architecture.
- `train.py` — Implements the Improved U-Net training and validation pipeline.
- `test.py` — Evaluates the Improved U-Net on the selected dataset split.
- `utils.py` — Provides shared evaluation utilities and segmentation metrics.
- `visual_failure.py` — Generates visualisations for analysing segmentation errors.
- `baseline_unet/modules.py` — Defines the baseline U-Net architecture.
- `baseline_unet/train.py` — Implements baseline U-Net training and validation.
- `baseline_unet/test.py` — Evaluates the baseline U-Net using the same test dataset.


## 3. Dataset and Preprocessing

### 3.1 Dataset Description

The HipMRI dataset contains grayscale MRI slices stored in NIfTI (.nii.gz) format, with two original spatial dimensions: 256 × 128 and 256 × 144 pixels. Of the 12,660 slices, 12,600 have dimensions of 256 × 128, while the remaining 60 have dimensions of 256 × 144.

To ensure consistent input dimensions during training and evaluation, the smaller slices were padded to 256 × 144. The corresponding segmentation masks were padded consistently to preserve spatial alignment between MRI images and their ground-truth annotations.

Each segmentation mask contains integer labels ranging from 0 to 5, representing six segmentation classes, including the background.


### 3.2 Dataset Splitting

The dataset is divided into training, validation and testing subsets. Patient-level separation is used to prevent MRI scans from the same patient appearing in multiple subsets and to avoid data leakage.

| Dataset Property | Training | Validation | Testing |
|---|---:|---:|---:|
| Number of MRI slices | 11,460 | 660 | 540 |
| Number of patients | 31 | 4 | 3 |

The dataset contains **12,660 MRI slices from 38 patients** across the three subsets.

Patient-level splitting is particularly important because the dataset contains longitudinal scans. Images collected from the same patient across different weeks may share anatomical characteristics, potentially introducing data leakage if distributed across training and evaluation subsets.

Patient identifiers were checked across the three subsets to verify that no patient appeared in more than one split. This ensures that the held-out test set evaluates model performance on patients not encountered during training.

### 3.3 Class Distribution and Imbalance

An analysis of the ground-truth segmentation masks revealed substantial class imbalance across the HipMRI dataset. While Classes 0 and 1 are present in every slice, Classes 4 and 5 occur considerably less frequently.

In the training set, Class 4 appears in only 35.39% of slices, while Class 5 appears in 36.23%. Similar distributions are observed in the validation and testing subsets, indicating that the imbalance persists across the dataset.

The pixel distribution is also uneven. Classes 1 and 2 occupy substantially more pixels than Classes 4 and 5, which represent comparatively small segmentation regions.

This imbalance presents a challenge for multi-class segmentation because models may learn more effectively from frequently represented classes. Consequently, overall segmentation metrics may conceal poorer performance on underrepresented structures.

To investigate this limitation, the project evaluates per-class Dice, IoU and boundary precision rather than relying exclusively on aggregate segmentation accuracy.

### 3.4 Data Preparation

Using the `nibabel` library, the MRI images and their corresponding segmentation masks are loaded. Each image is paired with its ground-truth mask by matching their filenames. This is done by replacing the `case_` prefix with `seg_`.

The preprocessing pipeline consists of the following steps:

**1. Intensity Normalisation** – Min-max normalisation is applied independently to each MRI slice, scaling intensity values to [0, 1]. Constant-intensity slices are replaced with zero-filled arrays to avoid division by zero.

**2. Spatial Padding** – Eight columns of zeros are added to both sides of the smaller images to standardise their dimensions from 256 × 128 to 256 × 144. Identical padding is applied to the segmentation masks using background label 0 to preserve spatial alignment.

**3. Tensor Conversion** – MRI images are converted into PyTorch tensors of shape `[1, 256, 144]`. Segmentation masks are converted into integer tensors of shape `[256, 144]`, preserving class labels 0–5 for use with multi-class cross-entropy loss.

## 4. Model Architecture

This project implements a 2D Improved U-Net architecture inspired by Isensee et al. (2018) [2] for multi-class segmentation of HipMRI images. The model adapts the reference architecture for 2D MRI slices while retaining its main structural components.

A baseline U-Net based on Ronneberger et al. (2015) [1] is implemented for comparison. The comparative study investigates whether the additional architectural components of the Improved U-Net provide meaningful improvements in segmentation accuracy, boundary reliability and computational efficiency.

### 4.1 Baseline U-Net

The baseline U-Net follows an encoder–decoder architecture with skip connections.

The encoder uses two consecutive 3 × 3 convolutional layers with ReLU activation at each level, followed by max pooling to progressively reduce spatial dimensions and extract features.

The decoder uses bilinear interpolation to upsample feature maps, which are concatenated with corresponding encoder features through skip connections. These combined features are processed using convolutional blocks to reconstruct spatial details.

Finally, a 1 × 1 convolution produces six output channels corresponding to the segmentation classes.

The baseline provides a reference for assessing whether the additional architectural complexity of the Improved U-Net delivers practical benefits.

### 4.2 Improved U-Net

The Improved U-Net is implemented as a 2D adaptation of the architecture proposed by Isensee et al. (2018) [2]. It retains the U-Net encoder–decoder structure while introducing residual context modules, strided convolutions, localisation modules and multi-resolution prediction fusion.

The architecture consists of the following components:

1. Encoder: Five encoding levels with 16, 32, 64, 128 and 256 feature channels. Residual context modules use convolutional layers, instance normalisation, LeakyReLU activation and dropout to extract features. Strided convolutions progressively reduce spatial resolution.

2. Decoder: Four decoding stages use nearest-neighbour upsampling followed by convolution. Features from the encoder are concatenated through skip connections and refined using localisation modules to reconstruct spatial information.

3. Multi-resolution Prediction Fusion: Segmentation predictions are generated at three decoder resolutions using 1 × 1 convolutions. Lower-resolution predictions are progressively upsampled and combined through element-wise addition to produce the final six-class segmentation logits.

## img 

The architecture is designed to incorporate contextual information from multiple spatial resolutions while maintaining a relatively lightweight model. Its effectiveness is assessed through segmentation accuracy, boundary precision and computational efficiency rather than architectural complexity alone.

### 4.3 Architecture Comparison

The following table summarises the main architectural differences between the baseline U-Net and the Improved U-Net.

| Component | Baseline U-Net | Improved U-Net |
|---|---|---|
| Architecture | 2D encoder–decoder | 2D encoder–decoder with residual context modules |
| Encoder | Double convolution blocks | Residual context modules |
| Downsampling | Max pooling | 3 × 3 strided convolutions |
| Activation | ReLU | LeakyReLU |
| Normalisation | No explicit normalisation | Instance normalisation |
| Dropout | Not implemented | Dropout (0.3) |
| Upsampling | Bilinear interpolation | Nearest-neighbour interpolation followed by convolution |
| Skip Connections | Encoder–decoder concatenation | Encoder–decoder concatenation |
| Decoder | Double convolution blocks | Localisation modules |
| Multi-resolution Prediction Fusion | Not implemented | Implemented |
| Output | Six-class segmentation logits | Six-class segmentation logits |

The models are primarily compared based on their architectural differences and how these differences influence their performance. Their effects on segmentation accuracy, boundary reliability and computational efficiency are examined in the experimental results.

## 5. Feasibility Review — HipMRI Improved U-Net

**Project:** Prostate Radiotherapy Planning | **Difficulty:** Normal | **Model:** 2D Improved U-Net

### 5.1 User Need, Scope and Acceptance Criteria

The prototype aims to assist imaging specialists by generating segmentation contours for prostate radiotherapy planning, subject to expert review. The main engineering question is whether the Improved U-Net offers meaningful improvements in segmentation accuracy and boundary reliability over the baseline at a reasonable computational cost.

**Acceptance Criteria:**

- Achieve prostate-specific test Dice ≥ 0.75, with an additional mean foreground Dice target of ≥ 0.90.
- Compare both models using per-class Dice, IoU and boundary precision on the same held-out test dataset.
- Analyse 3–5 difficult segmentation cases using ground-truth masks, predictions and error visualisations.
- Measure model complexity, peak GPU memory consumption and inference time per slice.

### 5.2 Model Choice and Course Concepts

The HipMRI dataset contains 11,460 training, 660 validation and 540 test slices, separated at the patient level.

The standard 2D U-Net serves as the baseline, while the Improved U-Net incorporates residual context modules, instance normalisation, learnable downsampling, localisation modules and multi-resolution prediction fusion, inspired by Isensee et al. (2018) [2].

Relevant concepts include CNNs, encoder–decoder segmentation, residual learning and regularisation. The 2D approach is computationally efficient but lacks explicit inter-slice context.

### 5.3 Preliminary Feasibility Evidence

Initial dataset inspection identified inconsistent image dimensions (256 × 128 and 256 × 144), resolved using symmetric zero-padding for images and masks. Patient-level separation between the training, validation and test sets was verified.

Both models were successfully implemented and trained. The Improved U-Net completed 20 epochs on Rangpur in **27 minutes and 38 seconds**, achieving a minimum validation loss of **0.0671 at epoch 4** and a mean foreground validation Dice of **0.8795**.

These findings established technical feasibility, but validation performance alone could not confirm the prostate-specific test Dice requirement or improvements over the baseline.

### 5.4 Risks, Computational Budget and Fallback

**Risks:** Class imbalance, missed anatomical regions, boundary inaccuracies and lack of explicit 3D context.

**Computational Budget:** Approximately 28 minutes per 20-epoch Improved U-Net training run, subject to GPU availability.

**Fallback:** Retain the baseline U-Net if the Improved U-Net does not provide sufficient segmentation or boundary improvements to justify its computational cost.

### 5.5 Experimental Investigation Plan

The following investigations were selected to assess the Improved U-Net:

- **Baseline Comparison:** Compare mean foreground Dice, IoU, boundary precision and per-class performance.
- **Architectural Ablation:** Replace baseline max pooling with strided convolutions to investigate learnable downsampling.
- **Failure Analysis:** Examine difficult cases for missed regions, false positives and boundary inaccuracies.
- **Computational Efficiency:** Compare trainable parameters, peak GPU memory and inference time per slice.
- **Engineering Assessment:** Assess segmentation reliability, computational trade-offs and requirements for practical deployment.

The implementation, results, observations and inferences are presented in subsequent sections.

## 6. Experimental Setup and Evaluation Metrics

### 6.1 Training Configuration

The baseline U-Net and Improved U-Net were implemented using PyTorch and trained for 20 epochs on the Rangpur HPC cluster using GPU acceleration.

Both models used Cross-Entropy Loss for multi-class segmentation. The validation dataset was used to monitor training and select checkpoints, while the held-out test dataset was reserved for final evaluation.

**Checkpoint Selection:** The checkpoint with the lowest validation loss was selected for each model rather than the final epoch. This criterion was used to favour generalisation while preventing test performance from influencing model selection.

### 6.2 Evaluation Procedure

Both architectures were evaluated on the same held-out test dataset, consisting of 540 MRI slices from three patients.

The evaluation considered overall segmentation performance, class-specific performance and computational efficiency to identify improvements and potential trade-offs.

The strided-convolution ablation model was evaluated using the same test dataset and metrics to investigate the contribution of learnable downsampling.

### 6.3 Evaluation Metrics

The following metrics were used to assess segmentation quality and computational efficiency.

| Metric | Purpose |
|---|---|
| **Dice Similarity Coefficient (DSC)** | Measures overlap between predicted and ground-truth regions. |
| **Intersection over Union (IoU)** | Measures overlap relative to the union of predicted and ground-truth regions. |
| **Boundary Precision (2-pixel tolerance)** | Measures the proportion of predicted boundary pixels within two pixels of the ground-truth boundary. |
| **Per-Class Dice and Boundary Precision** | Identifies performance differences across segmentation classes. |
| **Trainable Parameters** | Measures model size in terms of trainable parameters. |
| **Peak GPU Memory** | Measures GPU memory consumption during benchmarked evaluation. |
| **Inference Time** | Measures prediction latency per MRI slice. |

Mean foreground segmentation metrics exclude the background class (Class 0).

### 6.4 Experimental Comparisons

Three model configurations were evaluated:

1. **Baseline U-Net:** Standard 2D U-Net using max-pooling downsampling.
2. **Strided-Convolution Ablation:** Modified baseline U-Net with strided convolutions replacing max pooling.
3. **Improved U-Net:** 2D architecture incorporating residual context modules, strided downsampling, localisation modules and multi-resolution prediction fusion.

These comparisons assess whether the Improved U-Net's architectural changes provide sufficient improvements in segmentation quality and boundary reliability to justify their computational costs, helping determine its suitability for practical use.

## 7. Experimental Results and Analysis

This section evaluates whether the Improved U-Net provides sufficient improvements in segmentation quality, boundary reliability and computational efficiency to justify its selection over the baseline.

