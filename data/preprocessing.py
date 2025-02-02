import torch
import numpy as np
from torchvision import transforms

class AnimePreprocessor:
    def __init__(self, target_size=512):
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def process(self, image):
        """Process single PIL image"""
        return self.transform(image).unsqueeze(0)

    def process_batch(self, images):
        """Process batch of PIL images"""
        return torch.stack([self.transform(img) for img in images])
