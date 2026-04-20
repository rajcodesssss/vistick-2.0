import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from deep_translator import GoogleTranslator
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

# ✅ UPDATED GLOBAL CONTROL STATE
control = {
    "running": False,
    "language": "both",   # 🔥 ADDED
    "last_announcement": ""
}

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
    return {
        "status": "ok",
        "models": "loaded",
        "running": control["running"]
    }

@app.post("/control")
def set_control(state: int):
    if state not in [0, 1]:
        return JSONResponse(
            {"error": "state must be 0 or 1"},
            status_code=400
        )
    control["running"] = bool(state)
    status = "started" if state == 1 else "stopped"
    print(f"[Control] Pipeline {status} by app")
    return {
        "success": True,
        "running": control["running"],
        "status": status
    }

# ✅ UPDATED (LANGUAGE ADDED)
@app.get("/control/state")
def get_control_state():
    return {
        "running": control["running"],
        "language": control["language"],   # 🔥 ADDED
        "state": 1 if control["running"] else 0
    }

# ✅ NEW API
@app.post("/control/language")
def set_language(lang: str):
    if lang not in ["en", "hi", "both"]:
        return JSONResponse(
            {"error": "invalid language"},
            status_code=400
        )

    control["language"] = lang
    print(f"[Control] Language set to {lang}")

    return {
        "success": True,
        "language": lang
    }

@app.get("/status")
def get_status():
    return {
        "running": control["running"],
        "last_announcement": control["last_announcement"]
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not control["running"]:
        return JSONResponse(
            {"success": False, "error": "Pipeline stopped"},
            status_code=403
        )
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        bgr_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        detections = detector.detect(bgr_frame)
        scene_caption = captioner.caption(pil_image)
        announcement = fusion.build_announcement(detections, scene_caption)

        if announcement:
            announcement = announcement.replace(
                "describe this scene for a visually impaired person including surroundings, layout, and any hazards :",
                ""
            ).strip().lstrip(".,: ").strip()

        hindi = ""
        if announcement:
            try:
                hindi = GoogleTranslator(source="en", target="hi").translate(announcement)
                print(f"[Translation] ✓ {hindi}")
            except Exception as e:
                print(f"[Translation Error] {e}")
                hindi = ""

        if announcement:
            control["last_announcement"] = announcement

        print(f"[Analyze] EN: {announcement}")
        print(f"[Analyze] HI: {hindi}")

        return JSONResponse({
            "success": True,
            "announcement_en": announcement or "",
            "announcement_hi": hindi or "",
            "scene_caption": scene_caption,
            "detections": detections
        })

    except Exception as e:
        print(f"[Error] {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )

