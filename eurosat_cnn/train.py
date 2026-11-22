"""For training the EurosatCNN model the training dataset"""

import argparse
import os

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from common import DEVICE, prepare_dataset, TEST_SPLIT, TRAIN_SPLIT, RANDOM_SEED, BATCH_SIZE
from eurosat_cnn import EuroSATCNN, IMAGE_SIZE, N_CHANNELS

MODEL_OUTPUT_DIR = "eurosat_cnn/models/"
MODEL_SUMMARY_FILENAME = "model_summary.txt"
MODEL_VERSION_FILENAME = "model_version.json"

LEARNING_RATE = 0.001

EPOCHS = 32


def write_model_details(model: EuroSATCNN, iteration: int) -> None:
    """Writes a summary of the model to the provided path"""

    model_summary = summary(
        model, input_size=(1, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"]
    )
    with open(os.path.join(MODEL_OUTPUT_DIR, MODEL_SUMMARY_FILENAME), "w") as f:
        f.write(str(model_summary))

    model_version = {
        "iteration": iteration,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "model": str(model),
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "train_split": TRAIN_SPLIT,
        "test_split": TEST_SPLIT,
        "image_size": IMAGE_SIZE,
        "n_channels": N_CHANNELS,
    }
    with open(os.path.join(MODEL_OUTPUT_DIR, MODEL_VERSION_FILENAME), "w") as f:
        f.write(str(model_version))


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

    train, test = prepare_dataset()

    # train model
    trained_model = train_model(train)
    save_model(trained_model, output_path)
    write_model_details(trained_model, opt.iteration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a classification model on the EuroSAT dataset, of Sentinel-2 satellite images.")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration of the model to train")
    args = parser.parse_args()

    main(args)
