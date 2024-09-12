import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import VAE
from hyperparams import get_default_hyperparams

# Get hyperparameters and setup device
H = get_default_hyperparams()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained VAE model
vae = VAE(H).to(device)
vae.load_state_dict(torch.load("weights.pt", map_location=device))

# Define class names
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Create directories for each class if they don't exist
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)
for class_name in classes:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# Number of samples per class and total required
num_samples_per_class = 6000
batch_size = 5  # Number of samples per batch

def save_image(tensor, filepath):
    # If tensor is a PyTorch tensor, convert it to a NumPy array
    if isinstance(tensor, torch.Tensor):
        img = tensor.cpu().permute(1, 2, 0).numpy()  # Convert tensor to numpy and reorder dimensions
    elif isinstance(tensor, np.ndarray):
        img = tensor  # If it's already a NumPy array, just use it
    else:
        raise TypeError("Input should be a torch.Tensor or a numpy.ndarray")
    
    img = (img * 255).astype('uint8')  # Rescale image to [0, 255]
    
    # Save the image using the PIL library
    img_pil = Image.fromarray(img)
    img_pil.save(filepath)

# Generate images for each class
for i, class_name in enumerate(classes):
    print(f"Generating images for class: {class_name}")
    
    # Generate images in batches
    for sample_idx in range(0, num_samples_per_class, batch_size):
        # Randomly generate images as input to the VAE
        img = torch.rand(batch_size, 32, 32, 3).to(device)
        label = torch.LongTensor([i] * batch_size).to(device)  # Class label for all images in this batch
        
        # Reconstruct images using the VAE
        recs = vae.reconstruct(img, label, k=0)
        
        # Save each image in the corresponding class directory
        for j in range(batch_size):
            image_id = sample_idx + j + 1
            save_path = os.path.join(output_dir, class_name, f"{class_name}_{image_id:04d}.png")
            save_image(recs[j], save_path)

        print(f"Saved {sample_idx + batch_size} images for class: {class_name}")

print("Image generation complete!")
