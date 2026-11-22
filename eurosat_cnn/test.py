"""Evaluate a trained model against the test set."""

import argparse
import os

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall

from common import DEVICE, prepare_dataset
from eurosat_cnn import EuroSATCNN, NUM_CLASSES, INDEX_TO_CLASS
from image_classifier import EurosatImageClassifier

WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (0, 0, 255)
OVERLAY_SCALE = 5
BAND_HEIGHT = 80


def eval_model(model_path: str, test_dataloader: DataLoader[Tensor]) -> None:
    """Evaluates the model against the test set."""

    model = EuroSATCNN.from_file(model_path)

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


def create_overlays(model_path: str, overlay_path: str, test_dataloader: DataLoader[Tensor]) -> None:
    """Creates overlays of predictions"""

    image_classifier = EurosatImageClassifier(model_path)

    overall_index = 0
    correct = 0
    for data, labels in test_dataloader:
        data = data.to(DEVICE)
        results = image_classifier.predict(data)

        for labels_index, result in enumerate(results):
            overall_index += 1

            # Bring image back onto cpu, convert to numpy and convert back to 255 from normalised input frame.
            frame = data[labels_index].cpu().numpy() * 255
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            transposed_frame = frame.transpose(1, 2, 0)  # Tensors store the images in format (C, H, W) so transpose to (H, W, C) for opencv

            predicted_class = result.class_
            predicted_score = result.score
            ground_truth = INDEX_TO_CLASS[int(labels[labels_index])]

            if predicted_class == ground_truth:
                correct += 1

            image = cv2.resize(transposed_frame, None, fx=OVERLAY_SCALE, fy=OVERLAY_SCALE, interpolation=cv2.INTER_NEAREST)
            image = cv2.copyMakeBorder(image, 0, BAND_HEIGHT, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            prediction_colour = GREEN if predicted_class == ground_truth else RED
            lines = [
                (f"GT: {ground_truth}", WHITE),
                (f"Pred: {predicted_class}", prediction_colour),
                (f"Score: {predicted_score:.2%}", prediction_colour),
            ]

            band_top = image.shape[0] - BAND_HEIGHT
            for line_index, (text, colour) in enumerate(lines):
                y = band_top + 20 * (line_index + 1)
                cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

            cv2.imwrite(os.path.join(overlay_path, f"{overall_index:05d}_{ground_truth}_{predicted_class}.png"), image)

    # TODO: add precision and recall values for each class.
    print(f"Overlay images created at {overlay_path}. Overall accuracy: {round(correct / overall_index * 100, 2)}%")


def main(opt: argparse.Namespace) -> None:
    """Main function"""

    _, test = prepare_dataset()
    eval_model(opt.model_path, test)

    if opt.overlay_path is not None:
        create_overlays(opt.model_path, opt.overlay_path, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate a model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument("--overlay-path", type=str, default=None, help="Path to output overlays to")
    args = parser.parse_args()

    main(args)
