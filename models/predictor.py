# models/predictor.py
import torch
from models.unet import UNet
from config import PATCH_RADIUS, UNET_MODEL_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
model.eval()

CENTER = PATCH_RADIUS

def predict_pixel(image, i, j):
    patch = image[i-PATCH_RADIUS:i+PATCH_RADIUS,
                  j-PATCH_RADIUS:j+PATCH_RADIUS]

    patch = torch.tensor(patch).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_map = model(patch)
        pred = pred_map[0, 0, CENTER, CENTER]

    return int(pred.item())

