import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
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


def loader(ts, en):
    params=torch.load('./DM_model/models/model_{}_{}'.format(ts, en))
    model = SimpleCNN()
    model.load_state_dict(params)
    return model

def plot_conv1_filters(model, save_dir='./', filename='conv1_filters.png'):
    # Get the filters from the first convolutional layer (conv1)
    conv1_weights = model.conv1.weight.data.cpu().numpy()

    # Normalize filter values to the range [0, 1]
    min_wt = conv1_weights.min()
    max_wt = conv1_weights.max()
    conv1_weights = (conv1_weights - min_wt) / (max_wt - min_wt)

    # Number of filters
    num_filters = conv1_weights.shape[0]
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))  # Adjust grid size if needed

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # Convert the filter from (C, H, W) to (H, W, C) to display as an image
            img = np.transpose(conv1_weights[i], (1, 2, 0))

            # Plot the filter
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle('Filters Learned by Conv1 Layer', fontsize=16)
    
    # Save the plot to the specified directory
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory
    print(f'Plot saved at: {file_path}')


def plot_conv2_filters(model, save_dir='./', filename='conv2_filters.png'):
    # Get the filters from the second convolutional layer (conv2)
    conv2_weights = model.conv2.weight.data.cpu().numpy()

    # Normalize filter values to the range [0, 1]
    min_wt = conv2_weights.min()
    max_wt = conv2_weights.max()
    conv2_weights = (conv2_weights - min_wt) / (max_wt - min_wt)

    # Number of filters
    num_filters = conv2_weights.shape[0]
    
    # Prepare the plot - Adjust grid size depending on the number of filters
    num_cols = 8  # You can change the number of columns based on how you want the layout
    num_rows = num_filters // num_cols + (num_filters % num_cols > 0)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 2))

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # Convert the filter from (C, H, W) to (H, W, C) to display as an image
            # Filters in conv2 have 32 channels (since conv2 expects 32-channel inputs from conv1)
            img = np.transpose(conv2_weights[i], (1, 2, 0))

            # For visualization, we might only want to display the filter for a single channel
            # Let's display the filter for the first channel (you can experiment with others)
            img = img[:, :, 0]  # Display the filter for the first channel
            
            ax.imshow(img, cmap='gray')  # Use a grayscale colormap for easier viewing
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle('Filters Learned by Conv2 Layer (First Channel)', fontsize=16)
    
    # Save the plot to the specified directory
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory
    print(f'Plot saved at: {file_path}')


# Example usage:
date = '20240912_172513'
accuracy= '69'
model = loader(date, accuracy)
plot_conv1_filters(model, save_dir='./DM_model/plots/filters', filename=(f"conv1_filters{date}_{accuracy}.png"))
plot_conv2_filters(model, save_dir='./DM_model/plots/filters', filename=(f"conv2_filters{date}_{accuracy}.png"))