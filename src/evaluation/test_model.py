import torch
import pickle
import os
from model import GhostFaceNetV2

data_dir = "data/client1"
num_classes = len(os.listdir(data_dir))
model = GhostFaceNetV2(num_classes=num_classes)
with open("final_model.pth", "rb") as f:
    params = pickle.load(f)
state_dict = model.state_dict()
for i, (key, val) in enumerate(state_dict.items()):
    state_dict[key] = torch.tensor(params[i])
model.load_state_dict(state_dict)
print("Model loaded successfully")