# extract_pee.py
from models.predictor import predict_pixel

def extract_payload(stego, num_bits):
    recovered = stego.copy()
    bits = []
    idx = 0
    h, w = stego.shape

    for i in range(8, h - 8):
        for j in range(8, w - 8):
            if idx >= num_bits:
                return recovered, bits

            pred = predict_pixel(recovered, i, j)
            err = stego[i, j] - pred

            # Decode
            if err == 0 or err == 1:
                bits.append(err)
                recovered[i, j] = pred
                idx += 1
            elif err == -1 or err == -2:
                bits.append(-err - 1)
                recovered[i, j] = pred - 1
                idx += 1
            else:
                # reverse shifting
                if err > 1:
                    recovered[i, j] = stego[i, j] - 1
                elif err < -2:
                    recovered[i, j] = stego[i, j] + 1

    return recovered, bits

