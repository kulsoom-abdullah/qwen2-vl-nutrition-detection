from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, field_validator
from typing import List, Optional
import base64
from PIL import Image
import io
import os

from nutrition_detector.api.inference import InferenceEngine, MockInferenceEngine, VLLMInferenceEngine
from nutrition_detector.data.dataset import parse_bounding_boxes

app = FastAPI(title="Nutrition Table Detector API")

# Dependency injection configuration.
# The MockInferenceEngine allows for API testing without GPU resources.
# In production, this should be replaced with VLLMInferenceEngine via environment variables.
engine_type = os.getenv('INFERENCE_ENGINE', 'mock')
engine: InferenceEngine

if engine_type == 'vllm':
    # Assumption: MODEL_PATH env var is set if using vllm
    model_path = os.getenv('MODEL_PATH', 'Qwen/Qwen2-VL-7B-Instruct')
    engine = VLLMInferenceEngine(model_path=model_path)
else:
    engine = MockInferenceEngine()

class PredictionResponse(BaseModel):
    """Schema for the prediction response.

    Attributes:
        boxes: List of normalized bounding boxes [x_min, y_min, x_max, y_max].
        raw_text: The raw generation output from the vision-language model.
    """
    boxes: List[List[float]]
    raw_text: str

    @field_validator('boxes')
    @classmethod
    def validate_boxes(cls, v):
        for box in v:
            if len(box) != 4:
                raise ValueError(f"Each box must have 4 coordinates, got {len(box)}")
            for coord in box:
                if not (0.0 <= coord <= 1.0):
                    raise ValueError(f"Coordinates must be between 0.0 and 1.0, got {coord}")
        return v

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Detects nutrition tables in the provided image.

    Handles the full inference pipeline:
    1. Preprocessing: Resizes large images to max 1024px to prevent OOM/latency spikes.
    2. Inference: Calls the underlying inference engine (vLLM or Mock).
    3. Postprocessing: Parses the generated text into structured bounding box coordinates.

    Args:
        file: The uploaded image file (JPEG/PNG).

    Returns:
        JSON object containing a list of detected bounding boxes and the raw model output.

    Raises:
        HTTPException: If the file is not a valid image or inference fails.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    content = await file.read()
    
    try:
        # Client-side resizing logic moved to BFF layer to protect the inference engine
        # from massive payloads and ensure consistent input dimensions.
        image = Image.open(io.BytesIO(content))
        max_long_side = 1024
        if max(image.size) > max_long_side:
            image.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
        
        # Convert to base64 for transport to the inference backend (e.g., vLLM HTTP API)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        # Using a fixed prompt ensures consistent model behavior matching the fine-tuning distribution.
        prompt = "Detect all nutrition tables in this image and return the boxes."
        raw_text = engine.predict(img_base64, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Postprocessing extracts structured data from the unstructured LLM text generation.
    boxes = parse_bounding_boxes(raw_text)
    
    return {
        "boxes": boxes,
        "raw_text": raw_text
    }

@app.get("/health")
def health():
    """Health check endpoint for container orchestration (k8s/docker)."""
    return {"status": "ok"}
