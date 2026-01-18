# config.py

# -------------------------
# Dataset paths
# -------------------------
RAW_DATASET_DIR = "dataset/raw/"
TRAIN_DIR = "dataset/train/"
VAL_DIR = "dataset/val/"
TEST_DIR = "dataset/test/"

# -------------------------
# Model paths
# -------------------------
UNET_MODEL_PATH = "models/unet.pth"

# -------------------------
# RDH parameters
# -------------------------
PATCH_RADIUS = 8          # must match U-Net input (16x16 patch)
SKIP_BORDER = PATCH_RADIUS

# -------------------------
# Payload parameters
# -------------------------
DEFAULT_PAYLOAD_TEXT = "PatientID123"
PAYLOAD_SIZES_BITS = [32, 64, 128, 256, 512]

# -------------------------
# Training parameters
# -------------------------
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10

# -------------------------
# Results paths
# -------------------------
STEGO_DIR = "results/stego/"
RECOVERED_DIR = "results/recovered/"
TABLES_DIR = "results/tables/"
PLOTS_DIR = "results/plots/"

# -------------------------
# Misc
# -------------------------
RANDOM_SEED = 42
