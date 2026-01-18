from models.predictor import predict_pixel

def extract_payload(stego, num_bits):
    recovered = stego.copy()
    bits = []
    idx = 0
    h, w = stego.shape

    for i in range(8, h-8):
        for j in range(8, w-8):
            if idx >= num_bits:
                return recovered, bits
            pred = predict_pixel(recovered, i, j)
            new_err = stego[i, j] - pred
            bits.append(new_err % 2)
            recovered[i, j] = pred + (new_err // 2)
            idx += 1
    return recovered, bits
