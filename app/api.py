from fastapi import FastAPI, UploadFile, File,Form 

from PIL import Image
from io import BytesIO
from .inference import run_inference

app = FastAPI(title="GRPO Vision API")

@app.post("/caption")
async def caption_image(
    file: UploadFile = File(...),
    max_new_tokens: int = Form(30),
    num_beams: int = Form(3),
    repetition_penalty: float = Form(1.2),
    length_penalty: float = Form(1.0),
):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    result = run_inference(
        image,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
    )

    return result
@app.get("/health")
def health():
    return {"status": "ok"}
