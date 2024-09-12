import torch
import os
import torchvision.utils as vutils
from torchvision import transforms
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Function to create directories if they do not exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to load the model's state from a checkpoint
def load_generator(checkpoint_path, nz, nc, ngf, device):
    # Initialize the generator
    netG = Generator(nz, nc, ngf).to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the generator state dictionary from the checkpoint
    netG.load_state_dict(checkpoint['generator_state_dict'])
    
    # Set the generator to evaluation mode
    netG.eval()
    
    return netG

# Function to generate and save images
def generate_images(generator, nz, num_images, output_dir='./dataset_gan', device='cpu'):
    ensure_dir(output_dir)  # Ensure the output directory exists
    
    # Generate random noise as input for the generator
    noise = torch.randn(num_images, nz, 1, 1, device=device)
    
    # Generate fake images
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    
    # Save the generated images
    for i in range(num_images):
        vutils.save_image(fake_images[i], f"{output_dir}/gan_image_{i+1}.png", normalize=True)
        print(f"Saved gan_image_{i+1}.png")


if __name__ == "__main__":
    # Hyperparameters and settings
    nz = 100      # Size of the latent vector (same as used during training)
    nc = 3        # Number of channels in the generated images (3 for RGB)
    ngf = 64      # Size of feature maps in the generator (same as used during training)
    num_images = 60000  # Number of images to generate
    checkpoint_path = 'checkpoints/checkpoint_epoch_49.pth'  # Path to the checkpoint file
    output_dir = './dataset_gan'  # Directory where the generated images will be saved
    
    # Use CUDA if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained generator model from the checkpoint
    generator = load_generator(checkpoint_path, nz, nc, ngf, device)
    
    # Generate images
    generate_images(generator, nz, num_images, output_dir, device)
