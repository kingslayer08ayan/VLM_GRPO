import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from .rewards import total_reward

# --------------------
# Device
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# Model paths (Docker-safe)
# --------------------
BASE_MODEL = os.getenv("BASE_MODEL_PATH", "/models/base")
GRPO_MODEL = os.getenv("GRPO_MODEL_PATH", "/models/phase3_5_grpo")

print("🔹 Loading models...")
print(f"   BASE: {BASE_MODEL}")
print(f"   GRPO: {GRPO_MODEL}")

# --------------------
# Processor (shared)
# --------------------
processor = BlipProcessor.from_pretrained(GRPO_MODEL)

# --------------------
# Models
# --------------------
base_model = BlipForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16
).to(device).eval()

grpo_model = BlipForConditionalGeneration.from_pretrained(
    GRPO_MODEL,
    torch_dtype=torch.float16
).to(device).eval()

print("✅ Models loaded on", device)

# --------------------
# Caption generation
# --------------------
@torch.no_grad()
def generate_caption(
    model,
    image,
    max_new_tokens,
    num_beams,
    repetition_penalty,
    length_penalty,
):
    inputs = processor(images=image, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    text = processor.decode(output[0], skip_special_tokens=True)
    # 🔴 Sentence termination guard (Tier-1.5)
    if "." in text:
        text = text[: text.rfind(".") + 1]

    return text


# --------------------
# Main inference API
# --------------------
@torch.no_grad()
def run_inference(image: Image.Image) -> dict:
    base_caption = generate_caption(base_model, image)
    grpo_caption = generate_caption(grpo_model, image)

    # rewards → always reduce to scalar
    base_reward = total_reward(image, [base_caption]).mean().item()
    grpo_reward = total_reward(image, [grpo_caption]).mean().item()

    return {
        "base_caption": base_caption,
        "grpo_caption": grpo_caption,
        "base_reward": round(base_reward, 4),
        "grpo_reward": round(grpo_reward, 4),
        "delta": round(grpo_reward - base_reward, 4),
    }
