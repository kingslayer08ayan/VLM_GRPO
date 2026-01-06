# VLM-GRPO

Vision-Language Model fine-tuned using **Group Relative Policy Optimization (GRPO)**  
with **CLIP-based rewards**, deployed using **FastAPI + Streamlit**.

## Features
- BLIP base model
- SFT on Flickr 30k (tested with 1% data only for now)
- GRPO fine-tuning with KL regularization
- CLIP reward + length & repetition penalties
- Dockerized deployment
- Single-GPU inference
- Interactive Streamlit UI

## Quick Start (Docker)
- docker compose up --build
UI: http://localhost:8501
API: http://localhost:8000

## Training
python training/phase3_5_grpo.py
## Evaluation 
python training/phase3_eval.py

## Notes
- models are *not* stored in repo
- tested on single RTX 3050 GPU locally

