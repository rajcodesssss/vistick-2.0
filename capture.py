import cv2
import base64
import numpy as np
from PIL import Image

class FrameCapture:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        return frame, pil_image  # BGR for YOLO, PIL for BLIP

    def release(self):
        self.cap.release()