
#module 8 - mnist classification with pytorch
#main script for training using command-line arguments

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from utilities import ConvNet, training, plot


#read command-line arguments for model training
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MNIST Classification with PyTorch"
    )

    #set training epochs with a default of 2
    parser.add_argument(
        "-e", "--num_epochs",
        type=int, default=2,
        help="number of training epochs"
    )

    #set batch size with a default of 1000
    parser.add_argument(
        "-b", "--batch_size",
        type=int, default=1000,
        help="number of images per batch"
    )

    #set learning rate with a default of 0.001
    parser.add_argument(
        "-l", "--learning_rate",
        type=float, default=0.001,
        help="learning rate for the optimizer"
    )

    return parser.parse_args()


#load mnist data and train the neural network
def main():
    args = parse_arguments()

    #validate training parameters
    if args.num_epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    #select gpu when available or use cpu
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    #download and prepare mnist training and test datasets
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    #create data loaders using the selected batch size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    #initialize the convolutional neural network
    model = ConvNet().to(device)

    #configure loss function and adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    #train model and collect performance metrics
    model, loss_list, acc_train, acc_test = training(
        model,
        args.num_epochs,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device
    )

    #save training loss and accuracy plots
    plot(loss_list, acc_train, acc_test)
    print("Training complete! Plot saved as figure.png")


#execute training when the script is run directly
if __name__ == "__main__":
    main()
