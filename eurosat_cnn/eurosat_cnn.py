"""Simple CNN implemented in pytorch, trained on the EuroSAT dataset. Roughly follows this guide: https://www.datacamp.com/tutorial/pytorch-cnn-tutorial"""

import torch
from torch import nn, Tensor

from common import DEVICE

NUM_CLASSES = 10
N_CHANNELS = 3

IMAGE_SIZE = 64

INDEX_TO_CLASS = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake",
}


class EuroSATCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # The model will follow the following architecture
        # Batch size = N
        # Image channels = C0
        # N Classes = K

        #  1. Convolutional layer 1         in:NxC0x64x64 -> out:NxC1x64x64. C1=32, Kernel size=3, stride=1, padding=1, spatial dims remain the same
        #  2. Batch normalization layer 1   in:NxC1x64x64 -> out:NxC1x64x64. C1=32, spatial dims remain the same
        #  3. LeakyReLU Activation Function
        #  4. Max Pool                      in:NxC1x64x64 -> out:NxC1x32x32. Kernel size=2, stride=2, spatial dims halve

        #  4. Convolutional layer 2         in:NxC1x32x32 -> out:NxC2x32x32. C2=64, Kernel size=3, stride=1, padding=1 spatial dims remain the same
        #  5. Batch normalization layer 2   in:NxC2x32x32 -> out:NxC2x32x32. C2=64, spatial dims remain the same
        #  6. LeakyReLU Activation Function
        #  7. Max Pool                      in:NxC2x32x32 -> out:NxC2x16x16. Kernel size=2, stride=2, spatial dims halve

        #  8. Convolutional layer 3         in:NxC2x16x16 -> out:NxC3x16x16. C3=128, Kernel size=3, stride=1, spatial dims remain the same
        #  9. Batch normalization layer 3   in:NxC3x16x16 -> out:NxC3x16x16. C3=128, spatial dims remain the same
        #  10. LeakyReLU Activation Function
        #  11. Max Pool                      in:NxC3x16x16 -> out:NxC3x8x8. Kernel size=2, stride=2, spatial dims halve

        #  12. Convolutional layer 4         in:NxC3x8x8  -> out:NxC4x8x8. C4=256, Kernel size=3, stride=1, padding=1, spatial dims remain the same
        #  13. Batch normalization layer 4   in:NxC4x8x8  -> out:NxC4x8x8. C4=256, spatial dims remain the same
        #  14. LeakyReLU Activation Function
        #  15. Max Pool                      in:NxC4x8x8  -> out:NxC4x4x4. Kernel size=2, stride=2, spatial dims halve

        #  16. Global Average Pool           in:NxC4x4x4  -> out:NxC4
        #  17. Linear Layer                  in:NxC4      -> out:NxK (logits)
        self.conv1 = nn.Conv2d(in_channels=N_CHANNELS, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(num_features=256)
        self.leaky_relu = nn.LeakyReLU()
        self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(in_features=256, out_features=NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        """Executes a forward pass of the model."""

        # Conv pass 1
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)

        # Conv pass 2
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)

        # Conv pass 3
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)

        # Conv pass 4
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)

        # Head
        x = self.global_average_pool(x)
        x = self.flatten(x)
        x = self.linear_layer(x)

        return x

    def predict(self, x: Tensor) -> Tensor:
        """
        Runs inference on the provided images.

        Input:
            Accepts one or more images as an input tensor.
        Returns:
            Returns a Tensor containing the probability for each class.
        """

        self.eval()
        with torch.inference_mode():
            output = self.forward(x)
            return torch.nn.functional.softmax(output, dim=1)

    @classmethod
    def from_file(cls, file_path: str) -> "EuroSATCNN":
        """Loads the model from file"""

        model = cls()
        model.load_state_dict(torch.load(file_path))
        model.to(DEVICE)

        return model
