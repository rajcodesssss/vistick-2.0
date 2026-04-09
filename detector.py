from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def get_direction(self, x_center, frame_width):
        third = frame_width / 3
        if x_center < third:
            return "to your left"
        elif x_center > 2 * third:
            return "to your right"
        else:
            return "in front of you"

    def get_distance_hint(self, box_area, frame_area):
        ratio = box_area / frame_area
        if ratio > 0.25:
            return "very close"
        elif ratio > 0.08:
            return "nearby"
        elif ratio > 0.02:
            return "at a medium distance"
        else:
            return "far away"

    def detect(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        frame_area = h * w
        results = self.model(bgr_frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            if conf < 0.45:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) / 2
            box_area = (x2 - x1) * (y2 - y1)

            direction = self.get_direction(x_center, w)
            distance = self.get_distance_hint(box_area, frame_area)

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "direction": direction,
                "distance": distance,
                "bbox": [x1, y1, x2, y2]
            })

        # Sort by size (largest/closest first — most important for blind users)
        detections.sort(key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]), reverse=True)
        return detections[:5]  # Top 5 most relevant