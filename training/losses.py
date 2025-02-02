import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan'):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.loss = nn.MSELoss() if gan_mode == 'lsgan' else nn.BCEWithLogitsLoss()

    def get_labels(self, predictions, target_is_real):
        labels = self.real_label if target_is_real else self.fake_label
        return labels.expand_as(predictions)

    def __call__(self, predictions, target_is_real):
        labels = self.get_labels(predictions, target_is_real)
        return self.loss(predictions, labels)

class AnimeGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gan_loss = GANLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, fake_pred, real_pred, fake_images, real_images):
        g_loss = self.gan_loss(fake_pred, True)
        l1_loss = self.l1_loss(fake_images, real_images)
        return g_loss + 10 * l1_loss
