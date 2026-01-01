## Data Documentation
# Dataset Overview

This project uses the AT&T Faces Dataset (formerly ORL Faces), a widely used academic benchmark for face recognition research. The dataset contains grayscale facial images of multiple subjects captured under controlled conditions and is commonly employed for algorithm evaluation and comparison.

# Dataset Characteristics

Subjects: 40 individuals

Images per Subject: 10

Image Resolution: 112 × 112 (after preprocessing)

Image Type: Grayscale facial images

Variations: Facial expressions, lighting, and minor pose changes

# Data Usage in This Project

The dataset is used to simulate a smart campus federated learning scenario, where facial data is distributed across multiple clients (e.g., departments or campus nodes).

Each client holds a disjoint subset of subjects

No raw facial images are shared between clients or with the central server

Only model updates (weights) are exchanged during training

This setup reflects realistic privacy constraints in educational institutions.

# Data Directory Structure
data/
├── raw/          # Original dataset files (not tracked in Git)
├── processed/    # Preprocessed and client-partitioned data
└── README.md     # Dataset documentation

# Important Notes

The data/raw/ directory is excluded from version control via .gitignore

Raw datasets must be obtained independently by users

Preprocessed data is generated locally using provided scripts

# Preprocessing Pipeline

Data preprocessing is performed using scripts in src/utils/ and includes:

Face normalization and resizing to 112×112 pixels

Conversion to grayscale format

Partitioning of subjects into client-specific subsets

Preprocessing can be reproduced by running:

python src/utils/preprocess.py

# Ethical and Privacy Considerations

This project does not distribute or publish raw facial data

The dataset is used strictly for academic research and experimentation

Federated Learning is employed to demonstrate privacy-preserving model training

No personal or real-world campus data is used

This repository adheres to responsible AI and data-handling practices.

# Data Access

The AT&T Faces Dataset is publicly available for research purposes. Users must obtain the dataset directly from the original source and ensure compliance with its licensing and usage terms.

# Reproducibility Statement

All experiments in this repository can be reproduced without sharing raw facial images, reinforcing the privacy-preserving objectives of the project.
