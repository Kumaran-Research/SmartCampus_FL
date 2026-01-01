[Report_Mini_Project 1.docx](https://github.com/user-attachments/files/24399132/Report_Mini_Project.1.docx)# Smart Campus Federated Learning for Face Recognition
## 📌 Privacy-Preserving Face Recognition using Federated Learning

# 1. Problem Statement

Smart campus environments increasingly rely on face recognition systems for security, access control, and identity verification. Conventional centralized training approaches require aggregating facial data from multiple sources, which raises serious privacy, ethical, and regulatory concerns—especially in educational institutions handling sensitive student data. This project addresses the challenge of training an accurate face recognition system across distributed, privacy-sensitive datasets without centralizing raw data.

# 2. Research Motivation

Federated Learning (FL) enables collaborative model training while keeping data localized at client sites, thereby preserving privacy. This paradigm is particularly suitable for smart campus scenarios, where data sharing across departments or institutions is restricted. The project explores the feasibility of federated face recognition by simulating a multi-client campus environment using a standard benchmark dataset.

# 3. Method Overview

This work implements a federated face recognition pipeline using a lightweight convolutional neural network and a client–server learning framework.

Key components:

- Model: GhostFaceNetV2 (lightweight CNN for face recognition)

- Federated Framework: Flower (FL orchestration)

- Aggregation Strategy: FedProx to mitigate non-IID client data

- Preprocessing: Face normalization and resizing to 112×112 grayscale images

- Training: 4 communication rounds with 15 local epochs per client per round

# 4. System Architecture

The system follows a standard federated learning workflow:

- Each client preprocesses and trains on its local facial dataset

- Local model updates are sent to the central server

- The server aggregates updates using FedProx

- The updated global model is redistributed to clients

- Performance is evaluated across communication rounds

![Uploading Architecture Diagram.png…]()


Repository structure reflects this architecture:

- src/models/ — Neural network definitions

- src/training/ — Client and server training logic

- src/evaluation/ — Model verification and testing

- src/utils/ — Preprocessing and helper utilities

- configs/ — Centralized hyperparameters


# 5. Experimental Setup

- Dataset: AT&T Faces Dataset (40 subjects, 10 images per subject)

- Clients: 2 simulated clients with disjoint subject partitions

- Training Mode: CPU-based federated training

- Evaluation Metric: Client-side classification accuracy across rounds

# 6. Quantitative Results

Parsed accuracy metrics from federated training rounds are available in:

results/tables/experiment_results.csv


The results demonstrate stable convergence of the global model while preserving strict data locality.

# 7. Reproducibility

This repository is structured to ensure full reproducibility.

Setup
pip install -r requirements.txt

Data Preprocessing
python src/utils/preprocess.py

Federated Training
python src/training/server.py
python src/training/client.py data/client1
python src/training/client.py data/client2

Result Parsing
python scripts/parse_logs.py

# 8. Docker Deployment

For containerized execution, use the provided docker-compose.yml to run the full federated learning pipeline in isolated environments.

# 9. Future Work

Integration of differential privacy mechanisms

Scaling to larger client populations

Evaluation under stronger non-IID data distributions

Deployment in real-world smart campus infrastructure

# 10. References

McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data, AISTATS 2017



