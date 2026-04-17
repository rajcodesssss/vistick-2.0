import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager

from config import Config
from detector import ObjectDetector
from captioner import SceneCaptioner
from fusion import CaptionFusion

Config.validate()

print("[INIT] Loading models...")
detector = ObjectDetector()
captioner = SceneCaptioner()
fusion = CaptionFusion()
print("[INIT] Models ready")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SERVER] Ready for real-time streaming")
    yield

app = FastAPI(lifespan=lifespan)

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected")

    try:
        while True:
            data = await ws.receive_bytes()

            try:
                # Decode image
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # 🔥 IMPORTANT: reduce size (prevents crash)
                frame = cv2.resize(frame, (224, 224))

                # Detection
                detections = detector.detect(frame)

                # 🔥 TEMPORARY (stable mode)
                caption = "scene detected"

                # (Later you can enable BLIP again)
                # caption = captioner.caption(Image.fromarray(frame))

                result = fusion.build_announcement(detections, caption)

                await ws.send_text(result or "")

            except Exception as e:
                print("🔥 PROCESS ERROR:", e)
                await ws.send_text("error")

    except Exception as e:
        print("🔥 WS ERROR:", e)

    finally:
        print("[WS] Client disconnected")