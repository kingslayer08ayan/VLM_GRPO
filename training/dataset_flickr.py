# training/dataset_flickr.py
from datasets import load_dataset

def load_flickr(split="test[:2%]"):
    return load_dataset("lmms-lab/flickr30k", split=split)
