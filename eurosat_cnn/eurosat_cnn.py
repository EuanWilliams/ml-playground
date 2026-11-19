"""Simple CNN implemented in pytorch, trained on the EuroSAT dataset. Roughly follows this guide: https://www.datacamp.com/tutorial/pytorch-cnn-tutorial"""
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn

import torchvision
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

DATASET_DOWNLOAD_DIR = 'cnn/dataset/'

RANDOM_SEED = 530
RANDOM = torch.Generator().manual_seed(RANDOM_SEED)

# Params
BATCH_SIZE = 32

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
assert sum([TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT]) == 1.0

NUM_CLASSES = 10
N_CHANNELS = 3


def prepare_dataset() -> Tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
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


def show_sample(dataloader: DataLoader[torch.Tensor]) -> None:
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
        #  1. Convolutional layer 1         in:NxC0x64x64 -> out:NxC1x64x64. C1 = 32, Kernel size=3, stride=1, padding=1, spatial dims remain the same
        #  2. ReLU Activation Function
        #  3. Max Pool                      in:NxC1x64x64 -> out:NxC1x32x32. Kernel size = 2, stride 2, spatial dims reduce by 2
        #  4. Convolutional layer 2         in:NxC1x32x32 -> out:NxC2x32x32. C2 = 64, Kernel size = 3, stride 1, spatial dims remain the same
        #  5. ReLU Activation Function
        #  6. Max Pool                      in:NxC2x32x32 -> out:NxC2x16x16. Kernel size = 3, stride 1, spatial dims remain the same
        #  7. Global Average Pool           in:NxC2x16x16 -> out:NxC2
        #  8. Linear Layer                  in:NxC2       -> out:NxK (logits)
        self.conv1 = nn.Conv2d(N_CHANNELS, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(in_features=32, out_features=10)

        # TOOD: Finish

def main() -> None:
    """Main function"""

    train, val, test = prepare_dataset()
    show_sample(train)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("")
    # parser.add_argument()
    # args = parser.parse_args()

    main()
