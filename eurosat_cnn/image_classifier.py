"""Image classifier implementing EurosatCNN"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from torch import Tensor
from torchvision.io import decode_image, ImageReadMode

from common import DEVICE
from eurosat_cnn import EuroSATCNN, INDEX_TO_CLASS

DEFAULT_MODEL_PATH = "eurosat_cnn/models/eurosat_cnn_it_1/eurosat_cnn_it_1.pth"

IMAGE_TEST = "eurosat_cnn/dataset/eurosat/2750/HerbaceousVegetation/HerbaceousVegetation_5.jpg"

SCORE_PRECISION = 4
MIN_CLASS_CONFIDENCE = 0.01


@dataclass
class ClassificationResult:
    class_: str
    score: float
    all_results: Dict[str, float]  # class: score


class EurosatImageClassifier:
    def __init__(self, model_path: str) -> None:
        self.model = EuroSATCNN.from_file(model_path)

    def predict(self, x: Tensor) -> List[ClassificationResult]:
        """Runs prediction on the provided image(s)"""

        predictions = self.model.predict(x).cpu()

        results = []
        for prediction in predictions:
            result = np.argsort(-prediction).tolist()
            results.append(
                ClassificationResult(
                    class_=INDEX_TO_CLASS[result[0]],
                    score=round(float(prediction[result[0]]), SCORE_PRECISION),
                    all_results={
                        INDEX_TO_CLASS[index]: round(float(prediction[index]), SCORE_PRECISION)
                        for index in result
                        if float(prediction[index]) >= MIN_CLASS_CONFIDENCE
                    },
                )
            )

        return results


def main() -> None:
    """Main function"""

    img = decode_image(IMAGE_TEST, mode=ImageReadMode.RGB)  # uint8, (3, H, W), values 0–255
    img = img.float() / 255.0  # float32 in [0, 1]
    batch = img.unsqueeze(0)  # (1, 3, H, W) for a model
    batch = batch.to(DEVICE)

    classifier = EurosatImageClassifier(DEFAULT_MODEL_PATH)
    result = classifier.predict(batch)
    print(result)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("")
    # parser.add_argument()
    # args = parser.parse_args()
    #
    # main(args)

    main()
