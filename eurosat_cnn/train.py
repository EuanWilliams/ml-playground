"""For training the EurosatCNN model the training dataset"""

import argparse
import os

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import DEVICE, prepare_dataset
from eurosat_cnn import EuroSATCNN
from test import eval_model

MODEL_OUTPUT_DIR = "eurosat_cnn/models/"
ITERATION = 1

LEARNING_RATE = 0.001

EPOCHS = 32


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


def save_model(model: EuroSATCNN, output_path: str) -> None:
    """Saves the model to the provided path"""

    torch.save(model.state_dict(), output_path)
    print(f"Saved model to {output_path}")


def main(opt: argparse.Namespace) -> None:
    """Main function"""

    output_path = os.path.join(MODEL_OUTPUT_DIR, f"eurosat_cnn_it_{opt.iteration}.pth")
    if os.path.exists(output_path):
        raise ValueError(f"Output path already exists, please use a different iteration number, aborting training: {output_path}")

    train, _, test = prepare_dataset()

    # train model
    trained_model = train_model(train)
    save_model(trained_model, output_path)

    # test model
    eval_model(trained_model, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a classification model on the EuroSAT dataset, of Sentinel-2 satellite images.")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration of the model to train")
    args = parser.parse_args()

    main(args)
