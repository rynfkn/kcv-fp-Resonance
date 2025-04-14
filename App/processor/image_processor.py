import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

class ImageProcessor:
    """Processes image data and extracts embeddings"""
    def __init__(self, model_name):
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def process(self, image_path):
        """Extract image embeddings"""
        # Load and process the image
        image = Image.open(image_path).convert('RGB')
        inputs = self.image_processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # DINOv2 uses CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Create feature names and dataframe
        columns = [f'img_emb_{i}' for i in range(embedding.shape[1])]
        return pd.DataFrame(embedding, columns=columns)