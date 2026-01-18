# models/predictor.py
import torch
import numpy as np
from models.unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
model.load_state_dict(torch.load("models/unet.pth", map_location=device))
model.eval()

PATCH_RADIUS = 8

def predict_pixel(image, i, j):
    patch = image[i-PATCH_RADIUS:i+PATCH_RADIUS,
                  j-PATCH_RADIUS:j+PATCH_RADIUS]
    patch = torch.tensor(patch).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(patch)

    return int(pred.item())
