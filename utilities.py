
#Module 8 - mnist Classification with Pytorch
#supporting functions and convolutional neural network

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


#convolutional neural network for mnist classification
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        #first convolutional layer extracts image features
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #second convolutional layer extracts deeper features
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #fully connected layers produce digit predictions
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        #pass input through convolutional layers
        out = self.layer1(x)
        out = self.layer2(out)

        #flatten features for classification
        out = out.reshape(out.size(0), -1)

        #generate output scores for ten digit classes
        out = self.fc1(out)
        out = self.fc2(out)

        return out


#calculate model accuracy on a given dataset
def track_accuracy(model, loader, device):
    #switch model to evaluation mode
    model.eval()

    correct = 0
    total = 0

    #disable gradient calculations during evaluation
    with torch.no_grad():
        for images, labels in loader:
            #move images and labels to the selected device
            images = images.to(device)
            labels = labels.to(device)

            #generate predictions from the model
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            #count correct predictions and total samples
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #calculate accuracy as a percentage
    accuracy = 100 * correct / total

    #return model to training mode
    model.train()

    return accuracy



#train the model and track loss and accuracy
def training(model, num_epochs, train_loader, test_loader,
             criterion, optimizer, device):

    #store training loss and accuracy values
    loss_list = []
    acc_train = []
    acc_test = []

    #repeat training for the specified number of epochs
    for epoch in range(num_epochs):
        model.train()

        #create progress bar for training batches
        pbar = tqdm(
            train_loader,
            desc=f"Training: {epoch + 1:03d}/{num_epochs}",
            ncols=125,
            leave=True
        )

        running_loss = []

        #process each batch of training images
        for i, (images, labels) in enumerate(pbar, 1):
            images = images.to(device)
            labels = labels.to(device)

            #calculate predictions and loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())
            loss_list.append(loss.item())

            #update model weights using backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #display average training loss
            pbar.set_postfix(
                loss=f"{sum(running_loss) / i:10.6f}"
            )


        #evaluate accuracy on training and test datasets
        acc_train.append(track_accuracy(model, train_loader, device))
        acc_test.append(track_accuracy(model, test_loader, device))

    return model, loss_list, acc_train, acc_test



#plot training loss and model accuracy
def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
    #create figure for loss and accuracy curves
    plt.figure(figsize=(10, 4))

    #plot training loss over epochs
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    #plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(acc_train)
    plt.plot(acc_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend(['Training', 'Testing'])

    #adjust spacing and save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
