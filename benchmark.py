"""Evaluate a saved LingBot-MAP reconstruction artifact."""

import argparse
import json
import os

import numpy as np

from lingbot_map.evaluation import evaluate_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="Directory produced by demo.py --save_run")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--output", default=None, help="Optional JSON result path")
    args = parser.parse_args()
    with np.load(os.path.join(args.artifact, "predictions.npz")) as artifact:
        metrics = evaluate_artifact(
            {key: artifact[key] for key in artifact.files},
            roi=tuple(args.roi) if args.roi else None,
            conf_threshold=args.conf_threshold,
        )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(text + "\n")


if __name__ == "__main__":
    main()
