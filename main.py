import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from config import Config
from detector import ObjectDetector
from captioner import SceneCaptioner
from fusion import CaptionFusion

Config.validate()

print("[Step 1] Loading YOLOv8...")
detector = ObjectDetector()
print("[Step 1] YOLOv8 ready")

print("[Step 2] Loading BLIP...")
captioner = SceneCaptioner()
print("[Step 2] BLIP ready")

fusion = CaptionFusion()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] All models loaded — ready to serve")
    yield

app = FastAPI(
    title="Vistick 2.0 Server",
    description="Visual assistant backend for Raspberry Pi client",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    """Raspberry Pi calls this first to confirm server is ready."""
    return {"status": "ok", "models": "loaded"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Core endpoint.
    Receives JPEG frame from Raspberry Pi camera.
    Runs YOLOv8 + BLIP + Fusion.
    Returns plain announcement text — Pi speaks it locally.
    """
    try:
        # Read uploaded JPEG bytes
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Convert to BGR numpy for YOLO
        bgr_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Run detection + captioning in parallel would be ideal
        # but keeping sequential for simplicity
        detections = detector.detect(bgr_frame)
        scene_caption = captioner.caption(pil_image)
        announcement = fusion.build_announcement(detections, scene_caption)

        return JSONResponse({
            "success": True,
            "announcement": announcement or "",
            "scene_caption": scene_caption,
            "detections": detections
        })

    except Exception as e:
        print(f"[Error] {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)