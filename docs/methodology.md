# Methodology

## Federated Learning for Face Recognition

This project implements a federated learning system for face recognition using the AT&T Faces dataset. Federated learning allows training a shared model across multiple clients (e.g., edge devices or institutions) without exchanging raw data, preserving privacy.

### Approach

- **Model**: GhostFaceNetV2, a lightweight convolutional neural network optimized for face recognition tasks.
- **Framework**: Flower (flwr) for federated learning orchestration.
- **Strategy**: FedProx to handle non-IID data distributions across clients.
- **Preprocessing**: Face detection using OpenCV Haar cascades, followed by resizing to 112x112 grayscale images.
- **Training**: Clients train locally for 15 epochs per round, with proximal regularization to stabilize convergence.
- **Evaluation**: Accuracy computed on client-side test sets after each round.

### Key Components

- **Server**: Coordinates global model aggregation and communication rounds.
- **Clients**: Perform local training and evaluation on private datasets.
- **Data Split**: AT&T Faces divided into client-specific subsets for simulation.

This methodology ensures reproducible, privacy-preserving machine learning in distributed settings.
