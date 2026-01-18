# embed_pee.py
from models.predictor import predict_pixel

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

            # PEE embedding
            if err == 0:
                stego[i, j] = pred + bits[idx]
                idx += 1
            elif err == -1:
                stego[i, j] = pred - 1 - bits[idx]
                idx += 1
            else:
                # histogram shifting
                if err > 0:
                    stego[i, j] = image[i, j] + 1
                else:
                    stego[i, j] = image[i, j] - 1

    return stego, idx

