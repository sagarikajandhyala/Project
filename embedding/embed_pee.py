# embedding/embed_pee.py
from predictor import predict_pixel

MAX_ERR = 0          # controls reversibility vs capacity
HEADER_BITS = 32     # payload length header

def embed_payload(image, bits):
    stego = image.copy()
    idx = 0
    h, w = image.shape

    for i in range(8, h - 8):
        for j in range(8, w - 8):

            if idx >= len(bits):
                return stego, idx

            pred = predict_pixel(image, i, j)
            err = image[i, j] - pred

            # 🔐 HEADER MUST BE EMBEDDED ONLY IN PERFECTLY SAFE PIXELS
            if idx < HEADER_BITS and err != 0:
                continue

            # 🔒 PAYLOAD SAFETY CONSTRAINT
            if abs(err) > MAX_ERR:
                continue

            new_val = pred + (2 * err + bits[idx])

            # 🚨 OVERFLOW CHECK
            if new_val < 0 or new_val > 255:
                continue

            stego[i, j] = new_val
            idx += 1

    return stego, idx





