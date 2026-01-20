# training/train_unet.py
import torch
from models.unet import UNet
from dataloader import get_loader
import torch.nn.functional as F
from tqdm import tqdm
from config import EPOCHS

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_loader = get_loader("dataset/raw")

PATCH_RADIUS = 8  # 16x16 patch
CENTER = PATCH_RADIUS  # index of center pixel

for epoch in range(EPOCHS):
    total_loss = 0.0

    for x, y in tqdm(train_loader):
        x = x.to(device)              # (B,1,16,16)
        y = y.to(device)              # (B,1)

        pred_map = model(x)           # (B,1,16,16)

        # 🔑 extract predicted center pixel
        pred_center = pred_map[:, :, CENTER, CENTER]  # (B,1)

        loss = F.l1_loss(pred_center, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.6f}")

torch.save(model.state_dict(), "models/unet.pth")
print("✅ U-Net training complete. Model saved.")

