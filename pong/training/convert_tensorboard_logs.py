"""Convert tensorboard logs to one JSON file per scalar tag for easier analysis."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(input_dir: str, output_dir: str | None = None) -> None:
    """Converts tensorboard scalar logs to one JSON file per tag, grouped by run."""
    root = Path(input_dir)
    out = Path(output_dir) if output_dir else root / "tensorboard_json"
    out.mkdir(parents=True, exist_ok=True)

    data: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for f in root.rglob("events.out.tfevents.*"):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})  # 0 = keep all points, no downsampling
        ea.Reload()
        run = str(f.parent.relative_to(root))
        for tag in ea.Tags()["scalars"]:
            data[tag][run] += [{"step": e.step, "value": e.value, "wall_time": e.wall_time} for e in ea.Scalars(tag)]

    for tag, runs in data.items():
        for points in runs.values():
            points.sort(key=lambda p: p["step"])
        path = out / (re.sub(r"[^\w.-]+", "_", tag).strip("_") + ".json")  # "val/mAP@0.5" -> "val_mAP_0.5.json"
        path.write_text(json.dumps({"tag": tag, "runs": runs}, separators=(",", ":")))
        print(f"Wrote {path} ({sum(len(p) for p in runs.values())} points)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert tensorboard logs to one JSON file per scalar tag")
    parser.add_argument("--input", type=str, required=True, help="Path to the input tensorboard logs directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: <input>/tensorboard_json)")
    args = parser.parse_args()

    main(args.input, args.output)
