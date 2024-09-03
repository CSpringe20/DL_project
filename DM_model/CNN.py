import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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
        self.fc2 = nn.Linear(128, 2)  # Assuming 2 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def training():
    # Load datasets from the two directories
    dataset = datasets.ImageFolder(root='./DM datasets', transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [100000, 20000])
    train_loader = DataLoader(train_set, batch_size=4)
    val_loader = DataLoader(val_set, batch_size=4)
    print(dataset.classes)
    
    # Initialize the model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        
        running_loss = 0.
        last_loss = 0.

        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(images)

            # Compute the loss and its gradients
            loss = criterion(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_number * len(train_loader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.



        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for j, (vimages, vlabels) in enumerate(val_loader):
                voutputs = model(vimages)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (j + 1)
        print('LOSS train {} valid {}'.format(last_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : last_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = './models/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
        epoch_number += 1


def loader(ts, en):
    params=torch.load('./model_{}_{}'.format(ts, en))
    model = SimpleCNN()
    model.load_state_dict(params)
    return model

training()