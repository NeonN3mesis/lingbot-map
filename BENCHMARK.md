# Reconstruction benchmark

This benchmark exists to prevent visually worse changes from being promoted by
an unrelated improved metric. It evaluates saved raw model output; reconstruction
changes remain frozen until the measurements correctly rank known examples.

## Capture

```bash
python demo.py \
  --model_path checkpoints/lingbot-map.pt \
  --image_folder example/courthouse \
  --first_k 8 --mode streaming --depth_edge_threshold 0 \
  --save_run /tmp/lingbot-benchmark/courthouse-baseline-1 \
  --no_viewer
```

Repeat the identical command at least three times with a different output
directory. Each artifact contains raw predictions, uint8 preprocessed images,
source-frame hashes, the command, Git revision, and runtime identity.

## Evaluate

The courthouse façade ROI is provisional and intentionally manual. Automatic
plane selection would add another unvalidated algorithm.

```bash
python benchmark.py /tmp/lingbot-benchmark/courthouse-baseline-1 \
  --roi 120 40 480 260 \
  --output /tmp/lingbot-benchmark/courthouse-baseline-1/metrics.json
```

The independent measurements are:

- numerical validity (hard gate);
- local camera-space surface roughness;
- adjacent-frame 3D disagreement at tracked image features;
- robust signed-distance thickness inside the labelled façade ROI.

No composite score is calculated.

## Repeatability baseline (RX 7900 XTX)

Three identical eight-frame runs at commit `666c4a9` produced:

| Metric | Run 1 | Run 2 | Run 3 | Interpretation |
|---|---:|---:|---:|---|
| Finite/positive depth | 100% | 100% | 100% | Stable hard gate |
| Agreement median | 0.06312 | 0.06483 | 0.06342 | 2.7% range |
| Agreement p90 | 0.22431 | 0.21674 | 0.22073 | 3.4% range |
| Roughness p90 | 0.01372 | 0.01286 | 0.01390 | 7.8% range |
| Façade thickness | 0.13182 | 0.17505 | 0.13712 | 29.2% relative range |

Poses and intrinsics were bit-identical. Depth was identical at the median, but
the top 1% of absolute depth differences reached 12.8–27.9% of median depth.
The façade-thickness variance therefore reflects genuine ROCm output variance,
not evaluator randomness.

Requesting deterministic PyTorch algorithms selected a numerically corrupt path
(about 9% non-finite depth/confidence), so deterministic execution is unavailable
on the tested stack.

## Promotion rule

A candidate must be captured at least three times. It is rejected if any run
fails numerical validity. For the other metrics, improvement must exceed the
combined baseline and candidate ranges; medians alone are insufficient. No
percentage threshold will be invented until known visually better/worse frozen
artifacts have been ranked successfully on both courthouse and hallway scenes.
