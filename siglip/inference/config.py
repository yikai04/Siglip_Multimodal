"""Configuration for inference demo."""
import os
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Model checkpoint
CKPT_PATH = os.environ.get(
    "SIGLIP_CKPT",
    os.path.join(REPO_ROOT, "outputs", "exp21_dual_pretrained_v2", "best_siglip.pt"),
)

# Data directory (Flickr8k)
DATA_DIR = os.environ.get(
    "FLICKR8K_DIR",
    os.path.join(REPO_ROOT, "Flickr8k"),
)

# Device: auto-detect GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BF16: only enabled on GPU with Ampere+ capability (compute >= 8.0)
USE_BF16 = (
    DEVICE.type == "cuda"
    and torch.cuda.get_device_capability(DEVICE)[0] >= 8
)

# Max caption length (must match training config)
MAX_LEN = 64

# Image size
IMAGE_SIZE = 224

# Embedding dimension
EMBED_DIM = 768

# Number of unfrozen layers (must match Exp21 config)
NUM_UNFROZEN_VIT_LAYERS = 3
NUM_UNFROZEN_DB_LAYERS = 2

# Pretrained DistilBERT tokenizer path (local first)
DISTILBERT_LOCAL = os.path.join(REPO_ROOT, "pretrained_distilbert")

# Gradio server config
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7861

# Print config at startup
print(f"[Config] Device: {DEVICE}")
print(f"[Config] BF16: {USE_BF16}")
print(f"[Config] Ckpt: {CKPT_PATH}")
print(f"[Config] Data: {DATA_DIR}")