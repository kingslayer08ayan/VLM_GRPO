import torch
import json
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Using device: {DEVICE}")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "base")
OUT_FILE = os.path.join(ROOT, "data", "phase1_eval", "samples.json")

processor = BlipProcessor.from_pretrained(MODEL_DIR)
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

model.eval()

dataset = load_dataset("lmms-lab/flickr30k", split="test[:10]")

results = []

for i, sample in enumerate(dataset):
    image = sample["image"]
    gt = sample["caption"][0]

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)

    pred = processor.decode(out[0], skip_special_tokens=True)

    results.append({
        "id": i,
        "ground_truth": gt,
        "prediction": pred
    })

    print(f"\n[{i}]")
    print("GT :", gt)
    print("PRED:", pred)

os.makedirs("data/phase1_eval", exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Phase-1 evaluation saved to {OUT_FILE}")
