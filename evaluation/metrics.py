# metrics.py
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import numpy as np

def _get_data_range(img):
    """
    Automatically determine the data range for PSNR and SSIM.
    Works for both 0-255 and 0-1 images.
    """
    if img.dtype == np.uint8:
        return 255
    else:
        # For float images, assume normalized to [0,1]
        return 1.0

# Direct functions for import
def psnr(original, stego):
    """Compute PSNR between two images, auto-detecting data range."""
    data_range = _get_data_range(original)
    return sk_psnr(original, stego, data_range=data_range)

def ssim(original, stego):
    """Compute SSIM between two images, auto-detecting data range."""
    data_range = _get_data_range(original)
    return sk_ssim(original, stego, data_range=data_range, multichannel=True)

# Combined metrics function
def compute_metrics(original, stego):
    """Return both PSNR and SSIM as a tuple."""
    return psnr(original, stego), ssim(original, stego)

