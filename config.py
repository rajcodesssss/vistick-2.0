import os
from dotenv import load_dotenv

load_dotenv()  # loads from .env file

class Config:
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    
    # Add future keys here as your project grows
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    @classmethod
    def validate(cls):
        missing = []
        if not cls.HF_TOKEN:
            missing.append("HF_TOKEN")
        if missing:
            raise EnvironmentError(f"Missing required API keys: {', '.join(missing)}")

