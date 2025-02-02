from PIL import Image
import torch
import numpy as np

def tensor_to_image(tensor):
    """Convert torch tensor to PIL Image"""
    image = tensor.squeeze().cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) * 127.5
    return Image.fromarray(image.astype('uint8'))

def load_weights(model, weight_path):
    """Load model weights with safety check"""
    try:
        model.load_state_dict(torch.load(weight_path))
    except RuntimeError:
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    return model
