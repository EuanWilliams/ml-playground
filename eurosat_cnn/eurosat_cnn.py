"""Simple CNN implemented in pytorch, trained on the EuroSAT dataset. Roughly follows this guide: https://www.datacamp.com/tutorial/pytorch-cnn-tutorial"""

import os
from typing import Tuple

import argparse
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn, optim, Generator, Tensor
from torchmetrics import Accuracy, Precision, Recall
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms


DATASET_DOWNLOAD_DIR = "eurosat_cnn/dataset/"
MODEL_OUTPUT_DIR = "eurosat_cnn/models/"
ITERATION = 1

RANDOM_SEED = 530
RANDOM = Generator().manual_seed(RANDOM_SEED)

DEVICE = "mps"

# Params
BATCH_SIZE = 16

TRAIN_SPLIT = 0.84
VAL_SPLIT = 0.01
TEST_SPLIT = 0.15
assert sum([TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT]) == 1.0

LEARNING_RATE = 0.001

EPOCHS = 32

NUM_CLASSES = 10
N_CHANNELS = 3


def prepare_dataset() -> Tuple[DataLoader[Tensor], DataLoader[Tensor], DataLoader[Tensor]]:
    """Prepares the dataset, downlaods if needed, splits into test and train splits"""

    print("Downloading EuroSAT dataset.")
    dataset = EuroSAT(DATASET_DOWNLOAD_DIR, download=True, transform=transforms.ToTensor())
    print("Done downloading dataset.")

    print(f"Splitting dataset into train={TRAIN_SPLIT}, val={VAL_SPLIT}, test={TEST_SPLIT}")
    _train, _val, _test = random_split(dataset, [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT], generator=RANDOM)
    train_loader = DataLoader(_train, batch_size=BATCH_SIZE, shuffle=True, generator=RANDOM)
    val_loader = DataLoader(_val, batch_size=BATCH_SIZE, shuffle=True, generator=RANDOM)
    test_loader = DataLoader(_test, batch_size=BATCH_SIZE, shuffle=True, generator=RANDOM)
    return train_loader, val_loader, test_loader


def show_sample(dataloader: DataLoader[Tensor]) -> None:
    """Displays sample of dataset"""

    # get some random training images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    image = torchvision.utils.make_grid(images)

    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class EuroSATCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # The model will follow the following architecture
        # Batch size = N
        # Image channels = C0
        # N Classes = K
        #  1. Convolutional layer 1         in:NxC0x64x64 -> out:NxC1x64x64. C1=32, Kernel size=3, stride=1, padding=1, spatial dims remain the same
        #  2. ReLU Activation Function
        #  3. Max Pool                      in:NxC1x64x64 -> out:NxC1x32x32. Kernel size=2, stride=2, spatial dims halve

        #  4. Convolutional layer 2         in:NxC1x32x32 -> out:NxC2x32x32. C2=64, Kernel size=3, stride=1, spatial dims remain the same
        #  5. ReLU Activation Function
        #  6. Max Pool                      in:NxC2x32x32 -> out:NxC2x16x16. Kernel size=2, stride=2, spatial dims halve

        #  7. Global Average Pool           in:NxC2x16x16 -> out:NxC2
        #  8. Linear Layer                  in:NxC2       -> out:NxK (logits)
        self.conv1 = nn.Conv2d(in_channels=N_CHANNELS, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(in_features=64, out_features=10)

    def forward(self, input: Tensor) -> Tensor:
        """Executes a forward pass of the model."""

        # Conv pass 1
        x = self.conv1(input)
        x = self.relu(x)
        x = self.max_pool(x)

        # Conv pass 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # Head
        x = self.global_average_pool(x)
        x = self.flatten(x)
        x = self.linear_layer(x)

        return x


def train_model(train_dataloader: DataLoader[Tensor]) -> EuroSATCNN:
    """Trains the model"""

    model = EuroSATCNN().to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_index, (data, targets) in enumerate(tqdm(train_dataloader)):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)
            scores = model(data)
            loss = loss_function(scores, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(f"Finished epoch {epoch}, loss={loss}")

    return model


def eval_model(model: EuroSATCNN, test_dataloader: DataLoader[Tensor]) -> None:
    """Evaluates the model against the test set."""

    acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
    precision = Precision(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
    recall = Recall(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            acc(preds, labels)
            precision(preds, labels)
            recall(preds, labels)

    test_accuracy = acc.compute()
    test_precision = precision.compute()
    test_recall = recall.compute()
    print(f"Model achieved the following results on the test set: ")
    print(f"Accuracy: {round(float(test_accuracy) * 100, 2)}%")
    print(f"Precision: {round(float(test_precision) * 100, 2)}%")
    print(f"Recall: {round(float(test_recall) * 100, 2)}%")


def save_model(model: EuroSATCNN, output_path: str) -> None:
    """Saves the model to the provided path"""

    torch.save(model.state_dict(), output_path)


def main(opt: argparse.Namespace) -> None:
    """Main function"""

    output_path = os.path.join(MODEL_OUTPUT_DIR, f"eurosat_cnn_it_{opt.iteration}.pth")
    if os.path.exists(output_path):
        raise ValueError(f"Output path already exists, please use a different iteration number, aborting training: {output_path}")

    train, val, test = prepare_dataset()
    # show_sample(train)

    trained_model = train_model(train)
    save_model(trained_model, output_path)
    # model = EuroSATCNN()
    # model.load_state_dict(torch.load(os.path.join(MODEL_OUTPUT_DIR, f"eurosat_cnn_it_{ITERATION}.pth")))
    # model.to(DEVICE)
    eval_model(trained_model, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a classification model on the EuroSAT dataset, of Sentinel-2 satellite images.")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration of the model to train")
    args = parser.parse_args()

    main(args)
