import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from vae_model import VAE
import numpy as np

# Assuming your VAE class is named VAE and you have a function to load the model
# Define your VAE architecture and load the trained weights
model = VAE(latent_dim=20)  # Instantiate your VAE with the appropriate parameters
model.load_state_dict(torch.load('vae_cifar10.pt'))
model.eval()  # Set the model to evaluation mode

# Constants for normalization (assuming CIFAR-10)
mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])

# Generate random latent vectors
num_samples = 5
with torch.no_grad():
    z = torch.randn(num_samples, 20)  # Assuming latent_dim is 20

    # Decode latent vectors to images
    reconstructed_imgs = model.decode(z).cpu()

    # Denormalize the images (for visualization purposes)
    reconstructed_imgs = reconstructed_imgs * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    
    # Save the generated images
    save_image(reconstructed_imgs, 'generated_images.png', nrow=num_samples)

# Visualize the generated images using matplotlib
fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))

for i in range(num_samples):
    img = reconstructed_imgs[i].permute(1, 2, 0).numpy()  # permute to (H, W, C) for matplotlib, convert to numpy array
    img = np.clip(img, 0, 1)  # clip to ensure valid range before imshow
    axes[i].imshow(img)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
