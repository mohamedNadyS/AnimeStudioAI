import torch
from PIL import Image
from utils.image_processing import preprocess, postprocess

class AnimeConverter:
    """Main inference class for anime style conversion"""
    
    def __init__(self, style="ghibli", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.style = style
        self.model = self._load_model()
        self.model.eval()
        
    def _load_model(self):
        model = AnimeGANGenerator()
        model.load_state_dict(torch.load(f"weights/{self.style}_model.pth"))
        return model.to(self.device)
    
    def convert(self, input_image, output_size=512):
        """Convert input image to anime style"""
        # Preprocess
        tensor = preprocess(input_image, output_size).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(tensor)
            
        # Postprocess
        return postprocess(output)
    
    def batch_convert(self, input_paths, output_dir):
        """Batch process multiple images"""
        # Implement batch processing logic
        pass
