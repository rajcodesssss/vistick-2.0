import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import login
from config import Config

class SceneCaptioner:
    def __init__(self):
        # Authenticate with Hugging Face using token from config
        login(token=Config.HF_TOKEN)
        
        # Load BLIP large model — best quality for visually impaired context
        model_name = "Salesforce/blip-image-captioning-large"
        
        # Load processor — handles image preprocessing and tokenization
        self.processor = BlipProcessor.from_pretrained(model_name)
        
        # Load model — use float16 to save GPU memory on cloud server
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store device for later use during inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def caption(self, pil_image):
        # Conditional prompt biased toward visually impaired needs
        # Asks model to describe layout, surroundings and hazards
        prompt = "describe this scene for a visually impaired person including surroundings, layout, and any hazards:"
        
        # Preprocess image and prompt together, move to correct device
        inputs = self.processor(pil_image, prompt, return_tensors="pt").to(self.device)
        
        # Generate caption — num_beams=4 gives better quality than greedy
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode token ids back to readable string
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()