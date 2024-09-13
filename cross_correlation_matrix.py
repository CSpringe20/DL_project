import numpy as np
from scipy.signal import correlate2d
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
def loader(ts, en, model_type="vae"):
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

# Function to compute cross-correlation between two 2D feature maps
def compute_cross_correlation(feature_map_vae, feature_map_dm):
    # Check if the feature maps are 2D; if not, ensure you're extracting the 2D spatial dimensions
    if len(feature_map_vae.shape) != 2 or len(feature_map_dm.shape) != 2:
        raise ValueError('Feature maps must be 2D arrays for correlate2d')

    return correlate2d(feature_map_vae, feature_map_dm, mode='full')

# Function to find the best-matching filters between two models
def find_best_matching_filters(feature_maps_vae, feature_maps_dm):
    feature_maps_vae = feature_maps_vae.cpu().numpy()  # Remove batch dimension and convert to numpy
    feature_maps_dm = feature_maps_dm.cpu().numpy()

    num_filters_vae = feature_maps_vae.shape[1]  # Number of filters/channels in VAE model
    num_filters_dm = feature_maps_dm.shape[1]  # Number of filters/channels in DM model

    # Initialize a matrix to store cross-correlation values for all filter pairs
    cross_correlation_matrix = np.zeros((num_filters_vae, num_filters_dm))

    # Compute cross-correlation for every pair of filters
    for i in range(num_filters_vae):
        for j in range(num_filters_dm):
            # Extract the 2D spatial feature maps for the current filter pair
            vae_map_2d = feature_maps_vae[0, i, :, :]  # Batch index 0, filter i
            dm_map_2d = feature_maps_dm[0, j, :, :]    # Batch index 0, filter j
            
            cross_corr = compute_cross_correlation(vae_map_2d, dm_map_2d)
            cross_correlation_matrix[i, j] = np.max(np.abs(cross_corr))  # Store max cross-correlation

    # Find the best matching filter from the DM model for each VAE model filter
    best_matches = np.argmax(cross_correlation_matrix, axis=1)  # Get the index of the highest correlation
    max_correlation_values = np.max(cross_correlation_matrix, axis=1)  # Get the max correlation values

    return best_matches, max_correlation_values

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

# Ensure you obtain feature maps for both models before running the cross-correlation matrix calculation
# Load your trained models
model_vae = loader("20240913_090257", "97", model_type="vae")
model_dm = loader("20240912_214248", "70", model_type="dm")

# Load an input image and preprocess it
image = '4_6'
input_image = Image.open(f"./DM_model/DM_datasets/cifar10/{image}.png")  # Replace with your image path
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
input_image = transform(input_image)

# Get the feature maps from the first conv layer for both models
feature_maps_vae = get_conv1_feature_maps(model_vae, input_image)
feature_maps_dm = get_conv1_feature_maps(model_dm, input_image)

# Make sure to call find_best_matching_filters after obtaining feature maps
best_matches, max_correlation_values = find_best_matching_filters(feature_maps_vae, feature_maps_dm)

# Print the results
for i, (match_idx, max_corr) in enumerate(zip(best_matches, max_correlation_values)):
    print(f"VAE Filter {i} matches best with DM Filter {match_idx} with Max Cross-Correlation = {max_corr:.4f}")



import os

def plot_matching_filters_with_image(vae_filter, dm_filter, vae_idx, dm_idx, original_image, save_dir, image_name):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a 1x3 grid for 3 images

    # Plot original image
    axes[0].imshow(original_image.permute(1, 2, 0))  # Permute to (H, W, C) for display
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot VAE filter
    axes[1].imshow(vae_filter, cmap='gray')
    axes[1].set_title(f'VAE Filter {vae_idx}')
    axes[1].axis('off')

    # Plot DM filter
    axes[2].imshow(dm_filter, cmap='gray')
    axes[2].set_title(f'DM Filter {dm_idx}')
    axes[2].axis('off')

    # Save the figure to the specified directory
    save_path = os.path.join(save_dir, f"filters-{image_name}.png")
    plt.savefig(save_path)
    print(f"Plot saved as: {save_path}")

    plt.show()  # Show the plot after saving

# Find the filter pair with the maximum cross-correlation
max_corr_idx = np.argmax(max_correlation_values)  # Index of the VAE filter with the max correlation
dm_best_match_idx = best_matches[max_corr_idx]  # Corresponding DM filter index

# Print the filters and the corresponding correlation value
print(f"The VAE filter with the highest correlation is Filter {max_corr_idx}, which matches best with DM Filter {dm_best_match_idx}")
print(f"Max cross-correlation value: {max_correlation_values[max_corr_idx]}")

# Extract the corresponding feature maps for the best-matching filters
vae_filter_max_corr = feature_maps_vae[0][max_corr_idx].cpu().numpy()  # The VAE filter with the max correlation
dm_filter_max_corr = feature_maps_dm[0][dm_best_match_idx].cpu().numpy()  # The best-matching DM filter

# Directory to save the plots
save_dir = "./filter_comparison"

# Extract image name without extension
image_name = image.split('.')[0]

# Plot and save the matching filters along with the original image
plot_matching_filters_with_image(vae_filter_max_corr, dm_filter_max_corr, max_corr_idx, dm_best_match_idx, input_image, save_dir, image_name)
