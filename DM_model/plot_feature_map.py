import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image

# Define transformations for the training data
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a loader function to load model parameters based on the model type
def loader(ts, en, model_type="dm"):
    # Initialize the SimpleCNN model
    model = SimpleCNN()

    if model_type == "dm":
        # Load params from DM model
        params = torch.load('./DM_model/models/model_{}_{}'.format(ts, en))
    elif model_type == "vae":
        # Load params from VAE model
        params = torch.load('./vae_model/models/model_{}_{}'.format(ts, en))
    else:
        raise ValueError("Invalid model type. Choose 'dm' or 'vae'.")
    
    # Load the parameters into the model
    model.load_state_dict(params)
    return model

# Define a function to get the feature maps from conv1
def get_conv1_feature_maps(model, input_image):
    feature_maps = []  # List to store the feature maps
    
    # Register a hook to capture the output of conv1
    def hook_fn(module, input, output):
        feature_maps.append(output)
    
    # Register the hook to the conv1 layer
    hook = model.conv1.register_forward_hook(hook_fn)
    
    # Pass the input image through the model (up to conv1)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to calculate gradients for visualization
        _ = model(input_image.unsqueeze(0))  # Add batch dimension before passing input
    
    # Remove the hook after getting the feature maps
    hook.remove()
    
    return feature_maps[0]  # Return the captured feature maps

# Function to plot the feature maps
def plot_feature_maps(feature_maps, save_dir=None, filename=None):
    num_feature_maps = feature_maps.size(1)  # Get number of feature maps (channels)
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))  # Adjust grid size if needed

    # Normalize feature maps for better visualization
    feature_maps = feature_maps[0].cpu().numpy()  # Remove batch dimension and move to numpy
    feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min())

    for i, ax in enumerate(axes.flat):
        if i < num_feature_maps:
            ax.imshow(feature_maps[i], cmap='gray')  # Display the feature map as a grayscale image
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle('Feature Maps After Conv1 Layer', fontsize=16)
    
    if save_dir and filename:
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
        print(f"Feature maps saved at {os.path.join(save_dir, filename)}")
    else:
        plt.show()

# Load your trained model
model = loader("20240912_172513", "69", model_type="dm")

# Load an input image and preprocess it
image = 'cat_0994'
input_image = Image.open(f"./vae_model/VAE_datasets/vdvae/{image}.png")  # Replace with your image path
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
input_image = transform(input_image)

# Get the feature maps from the first conv layer
feature_maps = get_conv1_feature_maps(model, input_image)

# Plot and save the feature maps
plot_feature_maps(feature_maps, save_dir='./DM_model/plots/filters', filename=f"{image}_conv1_feature_maps.png")
