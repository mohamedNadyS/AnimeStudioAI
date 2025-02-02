import torch
import torch.nn as nn
from torchvision.models import vgg19

class StyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features
        for param in self.parameters():
            param.requires_grad_(False)

        self.style_layers = [1, 6, 11, 20]
        self.content_layers = [22]
        self.selected_layers = sorted(self.style_layers + self.content_layers)
        
        self.model = nn.Sequential()
        for i, layer in enumerate(self.vgg):
            if i in self.selected_layers:
                self.model.add_module(str(i), layer)
            if i == max(self.selected_layers):
                break

    def forward(self, x):
        features = []
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features
