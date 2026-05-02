import torch
import torch.nn as nn
from transformers import ViTModel
import numpy as np
from PIL import Image
import torchvision.transforms as T

# ELA logic
def apply_ela(image_path, quality=90):
    img = Image.open(image_path).convert("RGB")
    temp = "temp.jpg"
    img.save(temp, "JPEG", quality=quality)
    comp = Image.open(temp).convert("RGB")
    ela = Image.new("RGB", img.size)
    # Simple pixel-diff calculation
    ela = Image.fromarray(np.abs(np.array(img) - np.array(comp)) * 20)
    return img, ela

# ViT Model
class ViTForgeryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(768, 2) # 768 is hidden dim for base-patch16

    def forward(self, x):
        features = self.vit(x).last_hidden_state[:, 0, :]
        return self.fc(features)