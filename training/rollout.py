import torch

@torch.no_grad()
def generate_group(model, processor, image, K=4):
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
        num_return_sequences=K,
    )

    captions = processor.batch_decode(
        outputs, skip_special_tokens=True
    )

    return captions
