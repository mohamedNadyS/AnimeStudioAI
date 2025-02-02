import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import AnimeGANGenerator, AnimeGANDiscriminator
from data import AnimeDataset
from losses import GANLoss

def train(args):
    # Initialize datasets
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = AnimeDataset(args.dataset, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize models
    generator = AnimeGANGenerator().to(args.device)
    discriminator = AnimeGANDiscriminator().to(args.device)
    
    # Initialize optimizers
    g_optim = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Loss functions
    criterion = GANLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        for real_imgs in loader:
            real_imgs = real_imgs.to(args.device)
            
            # Train discriminator
            fake_imgs = generator(real_imgs)
            d_loss = criterion.compute_d_loss(
                discriminator, real_imgs, fake_imgs
            )
            
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            
            # Train generator
            g_loss = criterion.compute_g_loss(
                discriminator, fake_imgs, real_imgs
            )
            
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
        
        # Save checkpoints
        if epoch % args.save_interval == 0:
            torch.save(generator.state_dict(), f"checkpoints/generator_{epoch}.pth")
