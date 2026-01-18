# training/train_unet.py
import torch
from models.unet import UNet
from training.dataloader import get_loader
import torch.nn.functional as F
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_loader = get_loader("dataset/raw")

EPOCHS = 10   # enough for project-level results

for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = F.l1_loss(pred.squeeze(), y.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

torch.save(model.state_dict(), "models/unet.pth")
