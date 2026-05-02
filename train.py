import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor
import os
from PIL import Image

# 1. Dataset
class ForgeryDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, cls in enumerate(['authentic', 'tampered']):
            path = os.path.join(root_dir, cls)
            for f in os.listdir(path):
                self.samples.append((os.path.join(path, f), label))
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert('RGB')
        return self.transform(img), torch.tensor(label)

# 2. Model
class ViTModelFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(768, 2)
    def forward(self, x):
        return self.fc(self.vit(x).last_hidden_state[:, 0, :])

# 3. Train
dataset = ForgeryDataset('dataset')
loader = DataLoader(dataset, batch_size=2, shuffle=True)
model = ViTModelFT()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(10): # 10 epochs
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done.")

torch.save(model.state_dict(), "vit_forgery_finetuned.pth")
print("Training finished. Weights saved as vit_forgery_finetuned.pth")