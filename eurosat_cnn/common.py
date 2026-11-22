from typing import Tuple

import numpy as np
import torchvision
from matplotlib import pyplot as plt

from torch import Tensor
from torch import Generator
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import EuroSAT
from torchvision.transforms import transforms

DATASET_DOWNLOAD_DIR = "eurosat_cnn/dataset/"

TRAIN_SPLIT = 0.85
TEST_SPLIT = 0.15
assert sum([TRAIN_SPLIT, TEST_SPLIT]) == 1.0

RANDOM_SEED = 530
RANDOM = Generator().manual_seed(RANDOM_SEED)

BATCH_SIZE = 16

DEVICE = "mps"


def prepare_dataset() -> Tuple[DataLoader[Tensor], DataLoader[Tensor]]:
    """Prepares the dataset, downlaods if needed, splits into test and train splits"""

    print("Downloading EuroSAT dataset.")
    dataset = EuroSAT(DATASET_DOWNLOAD_DIR, download=True, transform=transforms.ToTensor())
    print("Done downloading dataset.")

    print(f"Splitting dataset into train={TRAIN_SPLIT} test={TEST_SPLIT}")
    _train, _test = random_split(dataset, [TRAIN_SPLIT, TEST_SPLIT], generator=RANDOM)
    train_loader = DataLoader(_train, batch_size=BATCH_SIZE, shuffle=True, generator=RANDOM)
    test_loader = DataLoader(_test, batch_size=BATCH_SIZE, shuffle=True, generator=RANDOM)

    return train_loader, test_loader


def show_sample(dataloader: DataLoader[Tensor]) -> None:
    """Displays sample of dataset"""

    # get some random training images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    image = torchvision.utils.make_grid(images)

    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
