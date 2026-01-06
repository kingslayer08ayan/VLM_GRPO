# training/phase3_eval.py

import json
import torch
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

from dataset_flickr import load_flickr
from rewards import total_reward

# =========================
# Config
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_ID = "Salesforce/blip-image-captioning-base"
GRPO_MODEL_DIR = "models/phase3_5_grpo"

EVAL_SPLIT = "test[:50]"          # small but meaningful
SAVE_PATH = "data/phase3_eval/results.json"

# =========================
# Load models
# =========================
print("Loading GRPO model...")
processor = BlipProcessor.from_pretrained(GRPO_MODEL_DIR)
grpo_model = BlipForConditionalGeneration.from_pretrained(
    GRPO_MODEL_DIR
).to(device).eval()

print("Loading base model...")
base_processor = BlipProcessor.from_pretrained(BASE_MODEL_ID)
base_model = BlipForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID
).to(device).eval()

# =========================
# Dataset
# =========================
dataset = load_flickr(EVAL_SPLIT)

# =========================
# Caption generation
# =========================
@torch.no_grad()
def generate_caption(model, processor, image):
    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=64,          # HARD CAP (important)
        do_sample=True,
        temperature=0.55,           # lower randomness
        top_p=0.8,
        repetition_penalty=1.35,
        no_repeat_ngram_size=3,
    )

    return processor.decode(output[0], skip_special_tokens=True)

# =========================
# Evaluation loop
# =========================
results = []
base_rewards = []
grpo_rewards = []

print("\n====== Phase-3 GRPO Evaluation ======\n")

for i in tqdm(range(len(dataset))):
    sample = dataset[i]
    image = sample["image"]
    gt = sample["caption"][0]

    pred_base = generate_caption(base_model, base_processor, image)
    pred_grpo = generate_caption(grpo_model, processor, image)

    r_base = total_reward(image, pred_base)
    r_grpo = total_reward(image, pred_grpo)
    r_base = r_base.mean().item()
    r_grpo = r_grpo.mean().item()

    base_rewards.append(r_base)
    grpo_rewards.append(r_grpo)

    results.append({
        "id": i,
        "ground_truth": gt,
        "base_caption": pred_base,
        "grpo_caption": pred_grpo,
        "base_reward": r_base,
        "grpo_reward": r_grpo,
    })

    # 🔍 print first few comparisons
    if i < 10:
        print(f"\n[{i}]")
        print("GT   :", gt)
        print("BASE :", pred_base)
        print("GRPO :", pred_grpo)
        rb = r_base
        rg = r_grpo
        print(f"Reward → BASE: {rb:.4f} | GRPO: {rg:.4f}")
        print("-" * 60)

# =========================
# Summary
# =========================
avg_base = sum(base_rewards) / len(base_rewards)
avg_grpo = sum(grpo_rewards) / len(grpo_rewards)

print("\n====== Reward Summary ======")
print(f"BASE avg reward : {avg_base:.4f}")
print(f"GRPO avg reward : {avg_grpo:.4f}")
print(f"Δ improvement   : {avg_grpo - avg_base:.4f}")

# =========================
# Save results
# =========================
with open(SAVE_PATH, "w") as f:
    json.dump(
        {
            "avg_base_reward": avg_base,
            "avg_grpo_reward": avg_grpo,
            "delta": avg_grpo - avg_base,
            "samples": results,
        },
        f,
        indent=2
    )

print(f"\n✅ Phase-3 evaluation saved to {SAVE_PATH}")
