import os
import torchvision
from torchvision import datasets, transforms
from PIL import Image

# Define the transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./dataa', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./dataa', train=False, download=True, transform=transform)

# Function to save images as PNG files
def save_images(dataset, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, (img, label) in enumerate(dataset):
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(folder, f'{i}_{label}.png'))

# Save training and test images
save_images(train_dataset, './cifar10')
save_images(test_dataset, './cifar10')

print("Images saved successfully!")