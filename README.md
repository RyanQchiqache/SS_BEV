# **SS\_BEV: Semantic Segmentation or BEV from Satellite Images for Urban and Off-Road Autonomous Driving**

## **Project Overview**

SS\_BEV (Semantic Segmentation for Bird’s Eye View) is a deep learning project aimed at performing **semantic segmentation** on satellite imagery to facilitate **urban and off-road autonomous driving** applications. The goal is to train and fine-tune the **Mask2Former** model to accurately segment aerial images into different classes, such as buildings, roads, vegetation, water, and land.

---

## **Motivation**

Semantic segmentation of satellite imagery is crucial for applications such as:

* **Autonomous Driving:** Understanding the environment from a top-down perspective.
* **Urban Planning:** Analyzing city layouts and identifying infrastructure.
* **Off-Road Navigation:** Mapping non-urban areas for autonomous vehicles.
* **Disaster Management:** Identifying affected areas from aerial data.

By leveraging **Mask2Former** (a state-of-the-art segmentation model), this project aims to generate high-quality segmented maps from aerial images.

---

## **Project Structure**

```
SS_BEV/
├── codeBase/                  # Main codebase for the project
│   ├── data/                  # Data handling and preprocessing
│   ├── models/                # Model definition and training scripts
│   ├── visualisation/         # Visualization utilities (e.g., overlays, comparisons)
│   │   ├── __init__.py
│   │   ├── config.yaml        # Configuration file for the project
│   │   └── main.py            # Main script to execute the pipeline
├── my_venv/                   # Virtual environment
├── SS_data/                   # Dataset folder containing images and masks
├── tests/                     # Unit tests for all components
│   ├── __init__.py
│   ├── test_data_preprocessor.py
│   └── test_satellite_dataset.py
├── LICENSE
├── README.md
├── requirements.txt
```

---
## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
