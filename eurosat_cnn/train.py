"""For training the EurosatCNN model the training dataset"""

import argparse
import os
import json
from datetime import datetime

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from common import DEVICE, prepare_dataset, TEST_SPLIT, TRAIN_SPLIT, RANDOM_SEED, BATCH_SIZE
from eurosat_cnn import EuroSATCNN, IMAGE_SIZE, N_CHANNELS

MODEL_OUTPUT_DIR = "eurosat_cnn/temp_models/"
MODEL_SUMMARY_FILENAME = "model_summary.txt"
MODEL_VERSION_FILENAME = "model_version.json"

LEARNING_RATE = 0.001

EPOCHS = 32


def write_model_details(model: EuroSATCNN, model_output_dir: str, iteration: int) -> None:
    """Writes a summary of the model to the provided path"""

    model_summary = summary(
        model, input_size=(1, N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=0
    )
    with open(os.path.join(model_output_dir, MODEL_SUMMARY_FILENAME), "w") as f:
        f.write(str(model_summary))

    model_version = {
        "iteration": iteration,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "train_split": TRAIN_SPLIT,
        "test_split": TEST_SPLIT,
        "image_size": IMAGE_SIZE,
        "n_channels": N_CHANNELS,
        "date_time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
    }
    with open(os.path.join(model_output_dir, MODEL_VERSION_FILENAME), "w") as f:
        f.write(json.dumps(model_version, indent=4))


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


def save_model(model: EuroSATCNN, output_dir: str, iteration: int) -> None:
    """Saves the model to the provided path"""

    model_output_path = os.path.join(output_dir, f"eurosat_cnn_it_{iteration}.pth")
    torch.save(model.state_dict(), model_output_path)
    print(f"Saved model to {model_output_path}")


def main(opt: argparse.Namespace) -> None:
    """Main function"""

    model_output_dir = os.path.join(MODEL_OUTPUT_DIR, f"eurosat_cnn_it_{opt.iteration}")
    if os.path.exists(model_output_dir):
        raise ValueError(f"Output path already exists, please use a different iteration number, aborting training: {model_output_dir}")

    os.makedirs(model_output_dir, exist_ok=True)

    train, test = prepare_dataset()

    # train model
    trained_model = train_model(train)
    save_model(trained_model, model_output_dir, opt.iteration)
    write_model_details(trained_model, model_output_dir, opt.iteration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a classification model on the EuroSAT dataset, of Sentinel-2 satellite images.")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration of the model to train")
    args = parser.parse_args()

    main(args)
