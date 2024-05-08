import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Any

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EndpointHandler():
    def __init__(self, path=""):
        self.processor = CLIPProcessor.from_pretrained("openai/openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/openai/clip-vit-large-patch14").to(device)
        self.model.eval()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        input_data = data.get("inputs", {})
        encoded_images = input_data.get("images")
        texts = input_data.get("texts", [])

        if not encoded_images or not texts:
            return {"error": "Both images and texts must be provided"} 

        try:
            images = [Image.open(BytesIO(base64.b64decode(img))).convert("RGB") for img in encoded_images]
            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

            # Move tensors to the same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                logits_per_text = outputs.logits_per_text  # this is the text-image similarity score

            return {
                "logits_per_image": logits_per_image.cpu().numpy().tolist(),
                "logits_per_text": logits_per_text.cpu().numpy().tolist()
            }
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return {"error": str(e)}
