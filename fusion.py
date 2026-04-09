class CaptionFusion:
    def __init__(self):
        self.last_announcement = ""

    def build_announcement(self, detections, scene_caption):
        parts = []

        # Scene overview from BLIP
        if scene_caption:
            parts.append(f"Scene: {scene_caption}.")

        # Object-level detail from YOLO with directions
        if detections:
            obj_parts = []
            for det in detections:
                label = det["label"]
                direction = det["direction"]
                distance = det["distance"]
                obj_parts.append(f"a {label} {distance} {direction}")

            objects_str = ", ".join(obj_parts)
            parts.append(f"I can see {objects_str}.")

        # Hazard warnings (priority objects)
        hazards = [d for d in detections if d["label"] in
                   ["person", "car", "truck", "bicycle", "motorcycle", "dog", "stairs", "fire hydrant"]]
        if hazards:
            hazard_names = [f"{h['label']} {h['direction']}" for h in hazards]
            parts.append(f"Caution: {', '.join(hazard_names)}.")

        announcement = " ".join(parts)

        # Avoid repeating the exact same announcement
        if announcement == self.last_announcement:
            return None
        self.last_announcement = announcement
        return announcement