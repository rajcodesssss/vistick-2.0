from gtts import gTTS
import os
import time

class TTSEngine:
    def __init__(self, output_dir="audio_output"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.latest_file = None

    def speak(self, text, lang="en"):
        if not text:
            return None
        filename = os.path.join(self.output_dir, f"announcement_{int(time.time())}.mp3")
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(filename)
        self.latest_file = filename
        return filename