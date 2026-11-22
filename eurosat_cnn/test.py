"""Evaluate a trained model against the test set."""

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, List, Dict, Counter as CounterT

import cv2
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from common import DEVICE, prepare_dataset, MODEL_FILE_EXTENSION
from eurosat_cnn import INDEX_TO_CLASS
from image_classifier import EurosatImageClassifier

WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (0, 0, 255)
OVERLAY_SCALE = 5
BAND_HEIGHT = 80

EVALUATION_DIR = "evaluation/"

OverlayData = Tuple[np.ndarray, str, str, float]  # Tuple of (image, ground_truth, predicted_class, score)


@dataclass
class ClassResult:
    class_: str
    precision: float
    recall: float
    accuracy: float


def find_model_file(model_dir: str) -> str:
    """Finds the model file in the provided directory"""

    for file in os.listdir(model_dir):
        if file.endswith(MODEL_FILE_EXTENSION):
            return os.path.join(model_dir, file)

    raise FileNotFoundError(f"No model file found in {model_dir} with extension {MODEL_FILE_EXTENSION}")


def create_overlays(overlays: List[OverlayData], evaluation_dir: str) -> None:
    """Creates overlays of predictions"""

    overlay_path = os.path.join(evaluation_dir, "overlays")
    os.makedirs(overlay_path, exist_ok=True)

    overall_index = 0

    for frame, ground_truth, predicted_class, score in overlays:
        # Bring image back onto cpu, convert to numpy and convert back to 255 from normalised input frame.
        frame = frame * 255
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        transposed_frame = frame.transpose(1, 2, 0)  # Tensors store the images in format (C, H, W) so transpose to (H, W, C) for opencv

        image = cv2.resize(transposed_frame, None, fx=OVERLAY_SCALE, fy=OVERLAY_SCALE, interpolation=cv2.INTER_NEAREST)
        image = cv2.copyMakeBorder(image, 0, BAND_HEIGHT, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        prediction_colour = GREEN if predicted_class == ground_truth else RED
        lines = [
            (f"GT: {ground_truth}", WHITE),
            (f"Pred: {predicted_class}", prediction_colour),
            (f"Score: {score:.2%}", prediction_colour),
        ]

        band_top = image.shape[0] - BAND_HEIGHT
        for line_index, (text, colour) in enumerate(lines):
            y = band_top + 20 * (line_index + 1)
            cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(overlay_path, f"{overall_index:05d}_{ground_truth}_{predicted_class}.png"), image)

        overall_index += 1

    print(f"Overlay images created at {overlay_path}.")


def write_class_metrics(class_metrics: List[ClassResult], model_dir: str) -> None:
    """Writes the class metrics to a file"""

    metrics_output_path = os.path.join(model_dir, EVALUATION_DIR, "class_metrics.json")
    output_json = {
        class_metric.class_: {"precision": class_metric.precision, "recall": class_metric.recall, "accuracy": class_metric.accuracy}
        for class_metric in class_metrics
    }
    with open(metrics_output_path, "w") as f:
        f.write(json.dumps(output_json, indent=4))


def calculate_metrics(class_results: Dict[str, CounterT[str]]) -> List[ClassResult]:
    """Calculates and prints precision, recall and accuracy for each class"""

    class_metrics: List[ClassResult] = []
    for class_name, results in class_results.items():
        tp = results["tp"]
        fp = results["fp"]
        fn = results["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        class_metrics.append(ClassResult(class_name, precision, recall, accuracy))

    return class_metrics


def eval_model(classifier: EurosatImageClassifier, test_dataloader: DataLoader[Tensor]) -> Tuple[Dict[str, CounterT[str]], List[OverlayData]]:
    """Evaluates the model against the test set"""

    # Counter of class: {tp, fp, fn}
    class_results: Dict[str, CounterT[str]] = {class_name: Counter() for class_name in INDEX_TO_CLASS.values()}

    overlays: List[OverlayData] = []  # List of tuples containing (image, ground_truth, predicted_class, score)

    for data, labels in test_dataloader:
        data = data.to(DEVICE)
        results = classifier.predict(data)

        for labels_index, result in enumerate(results):
            predicted_class = result.class_
            predicted_score = result.score
            ground_truth = INDEX_TO_CLASS[int(labels[labels_index])]

            if predicted_class == ground_truth:
                class_results[ground_truth]["tp"] += 1
            else:
                class_results[ground_truth]["fn"] += 1
                class_results[predicted_class]["fp"] += 1

            overlays.append((data[labels_index].cpu().numpy(), ground_truth, predicted_class, predicted_score))

    return class_results, overlays


def main(opt: argparse.Namespace) -> None:
    """Main function"""

    evaluation_dir = os.path.join(opt.model_dir, EVALUATION_DIR)
    os.makedirs(evaluation_dir, exist_ok=True)

    _, test = prepare_dataset()
    model_path = find_model_file(opt.model_dir)
    image_classifier = EurosatImageClassifier(model_path)

    class_results, overlays = eval_model(image_classifier, test)

    class_metrics = calculate_metrics(class_results)
    write_class_metrics(class_metrics, opt.model_dir)

    if opt.overlays is not None:
        create_overlays(overlays, evaluation_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate a model")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory of the model to evaluate")
    parser.add_argument("--overlays", action="store_true", help="Whether to create overlay images of the predictions")
    args = parser.parse_args()

    main(args)
