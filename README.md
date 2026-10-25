# module08_pytorch_mnist
MNIST Classification with PyTorch


# MNIST Classification with PyTorch

## Project Description
This project implements MNIST handwritten digit classification
using a convolutional neural network (CNN) in PyTorch.

The program is based on Module 7 Lecture B, Section 3 and uses
object-oriented programming (OOP) with command-line arguments
to configure model training.

## Project Files
- `run.py` - main script for training the model
- `utilities.py` - CNN model, training, accuracy, and plotting functions
- `requirements.txt` - required Python libraries
- `.gitignore` - excludes Python cache and downloaded data
- `figure.png` - generated training loss and accuracy plots
- `figure_default.png` - plot from training with default arguments

## Installation
Install the required libraries using:

```bash
python -m pip install -r requirements.txt
```

## Running the Program
Run with default parameters:

```bash
python run.py
```

Default values:
- num_epochs = 2
- batch_size = 1000
- learning_rate = 0.001

Run with custom parameters:

```bash
python run.py -e 1 -b 500 -l 0.0005
```

## Command-Line Arguments
- `-e`, `--num_epochs`: number of training epochs
- `-b`, `--batch_size`: batch size for training
- `-l`, `--learning_rate`: learning rate for the Adam optimizer

## Model and Results
The CNN uses convolutional layers, batch normalization,
ReLU activation, max pooling, and fully connected layers.

Training uses the MNIST dataset, cross-entropy loss,
and the Adam optimizer.

The program automatically downloads the MNIST dataset
and selects a GPU when available or the CPU otherwise.

After training, the program generates `figure.png`
containing training loss and training/testing accuracy curves.
