import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import yaml
from src.models.model import GhostFaceNetV2

class FaceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        for label, class_dir in enumerate(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_dir)
            for img_name in os.listdir(class_path):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(label)
        self.transform = lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0) / 255.0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

class FaceClient(fl.client.NumPyClient):
    def __init__(self, data_dir):
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        self.lr = config['client']['learning_rate']
        self.mu = config['client']['mu']
        self.batch_size = config['client']['batch_size']
        self.epochs = config['client']['epochs']
        self.device = torch.device("cpu")
        self.model = GhostFaceNetV2(num_classes=len(os.listdir(data_dir))).to(self.device)
        self.dataset = FaceDataset(data_dir)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for i, (key, val) in enumerate(state_dict.items()):
            state_dict[key] = torch.tensor(parameters[i])
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        print(f"Starting fit for {self.dataset.data_dir}")
        self.set_parameters(parameters)
        self.model.train()
        global_params = [torch.tensor(p) for p in parameters]
        for epoch in range(self.epochs):
            for images, labels in self.loader:
                print(f"Processing batch of {len(images)} images")
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if self.mu > 0:
                    for (name, param), global_param in zip(self.model.named_parameters(), global_params):
                        loss += (self.mu / 2) * torch.norm(param - global_param.to(self.device)) ** 2
                loss.backward()
                self.optimizer.step()
        print("Fit completed")
        return self.get_parameters(config), len(self.dataset), {}

    def evaluate(self, parameters, config):
        print(f"Starting evaluation for {self.dataset.data_dir}")
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Evaluation accuracy: {accuracy}")
        return 0.0, len(self.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1]
    client = FaceClient(data_dir)
    fl.client.start_client(server_address="localhost:8080", client=client)
