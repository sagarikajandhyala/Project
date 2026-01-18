# main.py
import os
import cv2
import numpy as np
import torch

from models.unet import UNet
from embedding.embed_pee import embed_payload
from extraction.extract_pee import extract_payload
from payload.utils import text_to_bits, bits_to_text
from evaluation.metrics import psnr, ssim


# ============================
# PATHS & SETTINGS
# ============================
DATASET_DIR = "dataset/raw"     # <-- YOUR DATASET
MODEL_PATH = "models/unet.pth"
MAX_IMAGES = 50                # limit for CPU runtime

# ============================
# LOAD TRAINED U-NET
# ============================
device = torch.device("cpu")

model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Phase-2 CNN predictor loaded (U-Net)")


# ============================
# PAYLOAD
# ============================
payload_text = "PatientID123"
payload_bits = text_to_bits(payload_text)

payload_len = len(payload_bits)
header_bits = [int(b) for b in format(payload_len, "032b")]
full_bits = header_bits + payload_bits


# ============================
# METRIC ACCUMULATORS
# ============================
psnr_list = []
ssim_list = []
success_count = 0
total_images = 0


# ============================
# LOAD IMAGE LIST
# ============================
image_files = sorted([
    f for f in os.listdir(DATASET_DIR)
    if f.endswith(".png")
])[:MAX_IMAGES]

print(f"📂 Images to process: {len(image_files)}")


# ============================
# MAIN PHASE-2 PIPELINE
# ============================
with torch.no_grad():  # IMPORTANT for speed + correctness

    for idx, fname in enumerate(image_files, start=1):
        print(f"▶ [{idx}/{len(image_files)}] Processing {fname}")

        img_path = os.path.join(DATASET_DIR, fname)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = image.astype(np.int32)
        total_images += 1

        # ----------------------------
        # EMBEDDING (CNN-BASED PEE)
        # ----------------------------
        stego, embedded_bits = embed_payload(image, full_bits)

        if embedded_bits < len(full_bits):
            print("  ⚠ Capacity insufficient, skipped")
            continue

        # ----------------------------
        # EXTRACTION (CNN-BASED)
        # ----------------------------
        recovered_img, extracted_bits = extract_payload(
    stego,          # original image
    embedded_bits
)


        # Decode header
        header_rec = extracted_bits[:32]
        rec_len = int("".join(map(str, header_rec)), 2)

        payload_rec_bits = extracted_bits[32:32 + rec_len]
        recovered_text = bits_to_text(payload_rec_bits)

        if recovered_text != payload_text:
            print("  ❌ Payload mismatch")
            continue

        # ----------------------------
        # METRICS
        # ----------------------------
        psnr_list.append(psnr(image, stego))
        ssim_list.append(ssim(image, stego))
        success_count += 1
        print("  ✅ Perfect recovery")


# ============================
# FINAL RESULTS
# ============================
print("\n================ RESULTS ================")
print("Total images processed:", total_images)
print("Successful reversible embeddings:", success_count)

if success_count > 0:
    print("Average PSNR:", sum(psnr_list) / len(psnr_list))
    print("Average SSIM:", sum(ssim_list) / len(ssim_list))
else:
    print("No successful embeddings.")
