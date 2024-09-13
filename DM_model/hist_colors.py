import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to load images from a folder and return them as numpy arrays
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            images.append(np.array(img))
    return np.array(images)

# Function to calculate color histograms for a dataset of images
def calculate_color_histograms(images):
    red_hist = np.zeros(256)
    green_hist = np.zeros(256)
    blue_hist = np.zeros(256)

    # Iterate over all images
    for img in images:
        # Flatten the image and calculate histogram for each channel
        red_hist += np.histogram(img[:, :, 0], bins=256, range=(0, 255))[0]
        green_hist += np.histogram(img[:, :, 1], bins=256, range=(0, 255))[0]
        blue_hist += np.histogram(img[:, :, 2], bins=256, range=(0, 255))[0]

    # Normalize histograms by total number of pixels
    total_pixels = images.shape[0] * images.shape[1] * images.shape[2]
    red_hist /= total_pixels
    green_hist /= total_pixels
    blue_hist /= total_pixels

    return red_hist, green_hist, blue_hist

# Paths to your folders
cifar10_folder = "./DM_model/DM_datasets/cifar10"
generated_folder = "./DM_model/DM_datasets/DM"

# Load images
cifar10_images = load_images_from_folder(cifar10_folder)
generated_images = load_images_from_folder(generated_folder)

# Calculate color histograms for both datasets
red_cifar, green_cifar, blue_cifar = calculate_color_histograms(cifar10_images)
red_gen, green_gen, blue_gen = calculate_color_histograms(generated_images)

# Plotting function
def plot_histograms(red_cifar, green_cifar, blue_cifar, red_gen, green_gen, blue_gen):
    plt.figure(figsize=(12, 6))

    # Plot Red Channel
    plt.subplot(1, 3, 1)
    plt.plot(red_cifar, label='CIFAR-10 Red', color='red')
    plt.plot(red_gen, label='Generated Red', color='darkred', linestyle='dashed')
    plt.title('Red Channel Distribution')
    plt.legend()

    # Plot Green Channel
    plt.subplot(1, 3, 2)
    plt.plot(green_cifar, label='CIFAR-10 Green', color='green')
    plt.plot(green_gen, label='Generated Green', color='darkgreen', linestyle='dashed')
    plt.title('Green Channel Distribution')
    plt.legend()

    # Plot Blue Channel
    plt.subplot(1, 3, 3)
    plt.plot(blue_cifar, label='CIFAR-10 Blue', color='blue')
    plt.plot(blue_gen, label='Generated Blue', color='darkblue', linestyle='dashed')
    plt.title('Blue Channel Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot the histograms for comparison
plot_histograms(red_cifar, green_cifar, blue_cifar, red_gen, green_gen, blue_gen)
