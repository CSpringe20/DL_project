import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0

    EPOCHS = 5

    losess = []
    all_preds = []
    all_labels = []

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
        running_vloss = 0
        with torch.no_grad():
            for vimages, vlabels in val_loader:
                voutputs = model(vimages)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
                _, predicted = torch.max(voutputs.data, 1)
                all_preds.extend(predicted.tolist())
                all_labels.extend(vlabels)
            avg_vloss = running_vloss / len(val_loader)
            print('LOSS train {} valid {}'.format(losess[-1], avg_vloss))
            epoch_number += 1
    cm = confusion_matrix(all_labels, all_preds)
    acc=(cm[0][0]+cm[1][1])*100/cm.sum()
    print('ACCURACY {}'.format(acc))
    model_path = './DM_model/models/model_{}_{}'.format(timestamp, int(acc))
    torch.save(model.state_dict(), model_path)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=dataset.classes)
    disp.plot()
    plt.savefig('./DM_model/plots/confusion_{}.png'.format(timestamp))
    plt.show()
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

def test(model):
    model.eval()
    with torch.no_grad():
        other_path="./vae_model/VAE datasets/vae_generated_dataset"
        tot=0
        gen=0
        progress_bar = tqdm(os.listdir(other_path), desc='Test', leave=True)
        for img_name in progress_bar:
            img_path = os.path.join(other_path, img_name)
            img = Image.open(img_path)
            img = transform(img)
            img = img.unsqueeze(0)  # Add batch dimension
            preds = model(img)
            _, predicted = torch.max(preds.data, 1)
            gen+=(predicted.item()==1)
            tot+=1
            progress_bar.set_postfix(acc=gen*100/tot)
        
training(100)
#test(loader("20240906", "214215_4"))