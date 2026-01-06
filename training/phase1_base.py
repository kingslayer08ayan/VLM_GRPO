# training/phase1_base.py

import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --------------------
# Paths
# --------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "base")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_ID = "Salesforce/blip-image-captioning-base"

# --------------------
# Load model
# --------------------
print("Loading BLIP base model...")

processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# --------------------
# Save locally
# --------------------
print(f"Saving model to {MODEL_DIR}")

processor.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)

print("✅ Phase 1 base model saved successfully.")
