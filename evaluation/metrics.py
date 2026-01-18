from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(original, stego):
    psnr = peak_signal_noise_ratio(original, stego, data_range=255)
    ssim = structural_similarity(original, stego, data_range=255)
    return psnr, ssim
