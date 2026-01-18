import cv2
import numpy as np
from embedding.embed_pee import embed_payload
from extraction.extract_pee import extract_payload
from payload.utils import text_to_bits, bits_to_text
from evaluation.metrics import compute_metrics

img = cv2.imread("dataset/raw/00000001_000.png", cv2.IMREAD_GRAYSCALE)
img = img.astype(int)

payload = "PatientID123"
bits = text_to_bits(payload)

stego, nbits = embed_payload(img, bits)
recovered, extracted_bits = extract_payload(stego, nbits)

psnr, ssim = compute_metrics(img, stego)

print("Recovered text:", bits_to_text(extracted_bits))
print("PSNR:", psnr)
print("SSIM:", ssim)
print("Perfect recovery:", np.array_equal(img, recovered))
