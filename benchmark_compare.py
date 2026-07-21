"""Summarize repeated benchmark metric files without hiding run variance."""

import argparse
import json

import numpy as np


def summarize(runs):
    summary = {}
    common = set.intersection(*(set(run) for run in runs))
    for key in sorted(common):
        values = [run[key] for run in runs]
        if not all(isinstance(value, (int, float)) and value is not None for value in values):
            continue
        array = np.asarray(values, dtype=float)
        mean = float(array.mean())
        summary[key] = {
            "values": [float(value) for value in array],
            "mean": mean,
            "min": float(array.min()),
            "max": float(array.max()),
            "relative_range": float((array.max() - array.min()) / max(abs(mean), 1e-12)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", help="metrics.json files from repeated runs")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    runs = []
    for path in args.metrics:
        with open(path, encoding="utf-8") as source:
            runs.append(json.load(source))
    text = json.dumps(summarize(runs), indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(text + "\n")


if __name__ == "__main__":
    main()
