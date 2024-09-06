import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

# Define transformations for the training data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def training(batch_size):
    # Load datasets from the two directories
    dataset = datasets.ImageFolder(root='./DM_model/DM datasets', transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [100000, 20000])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    # Initialize the model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0

    EPOCHS = 3

    best_vloss = 1_000_000.
    losess = []

    for epoch in range(EPOCHS):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        running_loss=0.
        tot=0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch_number + 1}', leave=True)
        for images, labels in progress_bar:
            tot+=1
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(images)

            # Compute the loss and its gradients
            loss = criterion(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            losess.append(loss.item())
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/tot)

        model.eval()

        correct = 0
        total = 0
        running_vloss = 0
        with torch.no_grad():
            for vimages, vlabels in val_loader:
                voutputs = model(vimages)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
                _, predicted = torch.max(voutputs.data, 1)
                total += vlabels.size(0)
                correct += (predicted == vlabels).sum().item()
            avg_vloss = running_vloss / len(val_loader)
            print('LOSS train {} valid {} ACC {}'.format(losess[-1], avg_vloss, correct/total))

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = './DM_model/models/model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)
            epoch_number += 1
    step=100000/batch_size
    plt.plot(losess)
    plt.xticks(np.arange(0, step*EPOCHS +1, step), map(str, np.arange(0, EPOCHS +1, 1)))
    plt.savefig('./DM_model/plots/model_{}.png'.format(timestamp))
    plt.show()


def loader(ts, en):
    params=torch.load('./DM_model/models/model_{}_{}'.format(ts, en))
    model = SimpleCNN()
    model.load_state_dict(params)
    return model

training(100)