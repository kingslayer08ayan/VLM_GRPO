import torch
import clip
from PIL import Image

device = "cuda"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

@torch.no_grad()
def clip_reward(image: Image.Image, captions):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize(captions).to(device)

    img_feat = clip_model.encode_image(image_input)
    txt_feat = clip_model.encode_text(text_input)

    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    return (img_feat @ txt_feat.T).squeeze(0)

def total_reward(image, captions):
    return clip_reward(image, captions)
