import torch
import clip
from PIL import Image

device = "cuda"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


@torch.no_grad()
def clip_reward(image: Image.Image, captions: list[str]):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize(captions).to(device)

    img_feat = clip_model.encode_image(image_input)
    txt_feat = clip_model.encode_text(text_input)

    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    return (img_feat @ txt_feat.T).squeeze(0)  # [K]


def length_penalty(captions, max_len=20):
    return torch.tensor(
        [-0.2 * max(0, len(c.split()) - max_len) for c in captions],
        device=device
    )

def sentence_end_bonus(captions):
    bonus = []
    for c in captions:
        c = c.strip()
        if c.endswith("."):
            bonus.append(0.2)
        else:
            bonus.append(-0.2)
    return torch.tensor(bonus, device=device)

def repetition_penalty(captions, n=3):
    penalties = []
    for text in captions:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        total = max(1, len(ngrams))
        unique = len(set(ngrams))
        ratio = unique / total
        penalties.append(-0.5 * min(1.0, (1 - ratio)))
    return torch.tensor(penalties, device=device)


def total_reward(image, captions):
    clip_r = clip_reward(image, captions).detach()
    len_r = length_penalty(captions).detach()
    rep_r = repetition_penalty(captions).detach()

    return (
        1.0 * clip_r
        + 0.4 * len_r
        + 0.8 * rep_r
        + 0.3 * sentence_end_bonus(captions)
    )
