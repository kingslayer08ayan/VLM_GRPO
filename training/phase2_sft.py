# training/phase2_sft.py
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset_flickr import load_flickr
from torch.amp import autocast
from tqdm import tqdm
def collate_fn(batch, processor, device):
    images = [b["image"] for b in batch]
    captions = [
        b["caption"][0] if isinstance(b["caption"], list) else b["caption"]
        for b in batch
    ]

    # Image encoding
    image_inputs = processor(
        images=images,
        return_tensors="pt"
    )

    # Text encoding (decoder inputs)
    text_inputs = processor.tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    input_ids = text_inputs["input_ids"]
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": image_inputs["pixel_values"].to(device),
        "input_ids": input_ids.to(device),
        "attention_mask": text_inputs["attention_mask"].to(device),
        "labels": labels.to(device),
    }




device = "cuda"

MODEL_ID = "Salesforce/blip-image-captioning-base"
OUT_DIR = "models/phase2_sft"

processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32
).to(device)

# 🔒 Freeze vision encoder
for p in model.vision_model.parameters():
    p.requires_grad = False

dataset = load_flickr("test[:2%]")
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, processor, device)
)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-6
)
model.train()

optimizer.zero_grad(set_to_none=True)

for step, batch in enumerate(loader):
    outputs = model(**batch)
    loss = outputs.loss

    if not torch.isfinite(loss):
        print(f"⚠️ NaN at step {step}, skipping")
        optimizer.zero_grad(set_to_none=True)
        continue

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if step % 20 == 0:
        print(f"Loss @ {step}: {loss.item():.3f}")



model.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)

model.eval()
sample = dataset[3]

inputs = processor(
    images=sample["image"],
    return_tensors="pt"
).to(device, torch.float32)

out = model.generate(**inputs, max_new_tokens=30)
print("GT :", sample["caption"][0])
print("PR :", processor.decode(out[0], skip_special_tokens=True))

