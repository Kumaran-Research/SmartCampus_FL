# Data Documentation

## Dataset Overview
This project uses the **AT&T Faces Dataset (ORL Faces)**, a standard academic benchmark widely used for face recognition research. The dataset consists of grayscale facial images collected under controlled conditions and is suitable for evaluating learning algorithms.

---

## Dataset Characteristics
- **Subjects:** 40 individuals  
- **Images per Subject:** 10  
- **Image Format:** Grayscale  
- **Preprocessed Resolution:** 112 × 112 pixels  
- **Variations:** Facial expressions, illumination changes, minor pose variations  

---

## Data Usage in This Project
The dataset is used to **simulate a smart campus federated learning environment**, where facial data is distributed across multiple clients (e.g., departments or campus nodes).

- Each client contains a **disjoint subset of subjects**
- **Raw facial images never leave the client**
- Only **model parameters** are shared with the central server
- This setup reflects real-world privacy constraints in educational institutions

---

## Directory Structure

data/

├── raw/          # Original dataset files (not tracked in Git)

├──client 1       # client-partitioned data 

├──client 2       # client-partitioned data 

├── processed/    # Preprocessed and client-partitioned data

└── README.md     # Dataset documentation


---

## Version Control Policy
- The `data/raw/` directory is **intentionally excluded** from version control via `.gitignore`
- Raw datasets must be downloaded independently by users
- No biometric or sensitive data is stored or shared through this repository

---

## Preprocessing Pipeline
Data preprocessing is performed using scripts in `src/utils/` and includes:
- Face normalization and resizing to 112×112 pixels
- Grayscale conversion
- Partitioning data into client-specific subsets

To reproduce preprocessing:

python src/utils/preprocess.py


## Ethical and Privacy Considerations

- This project does not distribute or publish raw facial data

-The dataset is used strictly for academic research and experimentation

-Federated Learning is employed to demonstrate privacy-preserving model training

-No personal or real-world campus data is used

-This repository adheres to responsible AI and data-handling practices.

## Data Access

The AT&T Faces Dataset is publicly available for research purposes. Users must obtain the dataset directly from the original source and ensure compliance with its licensing and usage terms.

## Reproducibility Statement

All experiments in this repository can be reproduced without sharing raw facial images, reinforcing the privacy-preserving objectives of the project.
