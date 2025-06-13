# SS_BEV: Semantic Segmentation and Bird’s Eye View (BEV) Generation from Satellite Imagery for Autonomous Driving

## Overview

**SS_BEV** is a research-grade deep learning framework for performing **semantic segmentation** on high-resolution **satellite imagery**, with the objective of generating **Bird’s Eye View (BEV)** representations for urban and off-road **autonomous driving** applications.

The pipeline integrates **Mask2Former**, a state-of-the-art universal segmentation model developed by Meta AI, and adapts it to aerial vision through a modular and extensible PyTorch-based architecture. This project delivers not only high-quality segmentation predictions but also tools for evaluation, visualization, and experimentation.

---

## Objectives

- Fine-tune and evaluate **Mask2Former** for dense pixel-wise classification on satellite images.
- Generate semantically rich BEV maps that serve as perception input for autonomous systems.
- Enable robust segmentation across diverse terrains including urban, suburban, and off-road areas.
- Provide visual interpretability, quantitative metrics (e.g., mean Intersection over Union), and modular tools for extension.

---

## Motivation

Semantic understanding of aerial data is a key enabler for multiple downstream tasks:

- **Autonomous Driving:** Enhances environmental perception from a top-down perspective, critical for planning and decision-making.
- **Urban Infrastructure Analysis:** Automates the detection of roads, buildings, vegetation, and other classes to support city planning.
- **Off-Road Navigation:** Facilitates the mapping of rural and undeveloped regions for autonomous vehicles.

By leveraging the generalization power of Mask2Former and incorporating a tailored data processing and training pipeline, this project bridges the gap between aerial vision and real-world autonomy requirements.

---



---

## **Project Structure**

```
SS_BEV/
├── codeBase/ # Core source code
│ ├── config/ # Logging setup and config loading
│ │ ├── config.yaml # Central configuration file for model, training, data, and paths
│ │ ├──  logging_setup.py # special logging file for user friendly logging
│ ├── data/ # Dataset loading, preprocessing, patchify/reconstruct logic
│ │ ├── DataPreprocessor.py
│ │ ├── satelite_dataset.py
│ ├── models/ # Model wrapper for Mask2Former training and evaluation
│ │ └── mask2former_model.py
│ ├── visualisation/ # Tools for visualizing segmentation results
│ │ └── visualizer.py
│ └── st_main.py # Entry point: initializes and runs the full segmentation pipeline
│
├── SS_data/ # Raw satellite image dataset (images/ and masks/) depending on your dataset
│ ├── images/
│ └── masks/
│
├── tests/ # Unit tests for individual components
│ ├── test_data_preprocessor.py
│ └── st_main_test.py
│
├── requirements.txt # Python package dependencies
├── LICENSE # MIT License
├── README.md # Project documentation
```

---

---

## Key Features

- **Data Preprocessing and Patchification:** Handles large satellite images by dividing them into fixed-size patches.
- **Flexible Albumentations Augmentation:** Configurable augmentation pipeline with flipping, rotation, brightness, blur, and cropping.
- **Configurable YAML Interface:** Easily tune training, augmentation, paths, and logging from a central config file.
- **Mixed Precision Training (AMP):** Optional automatic mixed precision for efficient training on modern GPUs.
- **TensorBoard Integration:** Real-time logging and visualization of training progress.
- **Quantitative Evaluation:** Supports mean IoU and per-class IoU metrics.
- **Full Reconstruction and Visualization:** Predicts and reconstructs entire images from patch-level outputs for qualitative inspection.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for detailed terms and conditions.

