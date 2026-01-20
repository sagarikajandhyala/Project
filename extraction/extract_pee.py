# extraction/extract_pee.py
from predictor import predict_pixel

def extract_payload(stego, original, num_bits):
    recovered = stego.copy()
    bits = []
    idx = 0
    h, w = stego.shape

    for i in range(8, h - 8):
        for j in range(8, w - 8):

            if idx >= num_bits:
                return recovered, bits

            # 🔒 IMPORTANT: predict ONLY from ORIGINAL image
            pred = predict_pixel(original, i, j)

            new_err = stego[i, j] - pred
            bits.append(new_err & 1)

            recovered[i, j] = pred + (new_err >> 1)
            idx += 1

    return recovered, bits






