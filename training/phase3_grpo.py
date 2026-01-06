import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from dataset_flickr import load_flickr
from rollout import generate_group
from rewards import total_reward

device = "cuda"
MODEL_ID = "models/phase2_sft"
OUT_DIR = "models/phase3_grpo"
ref_model = BlipForConditionalGeneration.from_pretrained(
    "models/phase2_sft"
).to(device)
ref_model.eval()
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_ID
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

dataset = load_flickr("test[:1%]")  # small & safe
model.train()

for step, sample in enumerate(dataset):
    image = sample["image"]
    captions = generate_group(model, processor, image, K=4)
    rewards = total_reward(image, captions)
    advantages = (rewards - rewards.mean()).detach()
    loss = 0
    for cap, adv in zip(captions, advantages):
        inputs = processor(
            images=image,
            text=cap,
            return_tensors="pt"
        ).to(device)
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        out = model(**inputs, labels=labels)
        with torch.no_grad():
            ref_out = ref_model(**inputs, labels=labels)
        loss += -adv * (out.loss - ref_out.loss)

    loss = loss / len(captions)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    if step == 100:
        break
    if step % 10 == 0:
        print(f"[GRPO] step {step} | loss {loss.item():.4f}")

model.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)
