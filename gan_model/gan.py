import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the Generator for BigGAN
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

# Define the Discriminator for BigGAN
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Hyperparameters for BigGAN training on CIFAR-10
nz = 100  # Size of latent vector (input to generator)
nc = 3    # Number of channels in the training images (CIFAR-10 has 3 channels: RGB)
ngf = 64  # Size of feature maps in the generator
ndf = 64  # Size of feature maps in the discriminator

# Initialize the generator and discriminator
netG = Generator(nz, nc, ngf).cuda()
netD = Discriminator(nc, ndf).cuda()

# Loss and optimizer
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1).cuda()  # Fixed noise vector to visualize progress
real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load and preprocess CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(trainset, batch_size=128, shuffle=True)

# Function to create directories if not present
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to save the model's state
def save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, checkpoint_dir='./checkpoints'):
    ensure_dir(checkpoint_dir)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': netG.state_dict(),
        'discriminator_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

# Function to train BigGAN on CIFAR-10
def train_GAN(num_epochs=25, checkpoint_dir='./checkpoints', image_dir='./images'):
    ensure_dir(checkpoint_dir)  # Ensure the checkpoint directory exists
    ensure_dir(image_dir)       # Ensure the image directory exists

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with real images
            netD.zero_grad()
            real_cpu = data[0].cuda()
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device='cuda')
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake images
            noise = torch.randn(b_size, nz, 1, 1, device='cuda')
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # Fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Step [{i}/{len(dataloader)}] Loss_D: {errD.item()} Loss_G: {errG.item()} D(x): {D_x} D(G(z)): {D_G_z1}/{D_G_z2}')

        # Save generated samples after each epoch for visualization
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        torchvision.utils.save_image(fake, f'{image_dir}/fake_samples_epoch_{epoch}.png', normalize=True)

        # Save model checkpoints after each epoch
        save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, checkpoint_dir)

# Call the training function to start training BigGAN
train_GAN(num_epochs=50)
