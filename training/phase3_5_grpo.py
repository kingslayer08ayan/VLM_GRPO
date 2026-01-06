# training/phase3_5_grpo.py
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from dataset_flickr import load_flickr
from rollout import generate_group
from rewards import total_reward

device = "cuda"

# 🔁 CONTINUE FROM PHASE-3
MODEL_ID = "models/phase3_grpo"
REF_MODEL_ID = "models/phase2_sft"
OUT_DIR = "models/phase3_5_grpo"

print("🔹 Loading models...")
processor = BlipProcessor.from_pretrained(MODEL_ID)

model = BlipForConditionalGeneration.from_pretrained(
    MODEL_ID
).to(device).train()

ref_model = BlipForConditionalGeneration.from_pretrained(
    REF_MODEL_ID
).to(device).eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

dataset = load_flickr("test[:1%]")  # small & safe

for step, sample in enumerate(dataset):
    image = sample["image"]

    # 🎲 Sample group
    captions = generate_group(model, processor, image, K=4)

    # 🎯 Rewards (already includes repetition penalty)
    rewards = total_reward(image, captions)
    advantages = (rewards - rewards.mean()).detach()

    loss = 0.0
    for cap, adv in zip(captions, advantages):
        inputs = processor(
            images=image,
            text=cap,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        out = model(**inputs, labels=labels)

        with torch.no_grad():
            ref_out = ref_model(**inputs, labels=labels)

        # 🔒 KL-anchored GRPO loss
        loss += adv * (out.loss - ref_out.loss)

    loss = loss / len(captions)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if step % 10 == 0:
        print(f"[GRPO-3.5] step {step} | loss {loss.item():.4f}")

    if step == 30:
        break

print("💾 Saving Phase-3.5 model...")
model.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)
