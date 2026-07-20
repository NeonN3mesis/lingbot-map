"""LingBot-MAP demo: streaming 3D reconstruction from images or video.

Usage:
    # Streaming inference (frame-by-frame with KV cache)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/

    # Streaming inference with keyframe KV caching
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/ --mode streaming --keyframe_interval 6

    # Windowed inference (for very long sequences, >500 frames)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10 --mode windowed --window_size 64

    # From video with custom FPS sampling
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10
"""

import argparse
import glob
import os
import sys
import tempfile
import time

# Must be set before `import torch` / any CUDA init. Reduces the reserved-vs-allocated
# memory gap by letting the caching allocator grow segments on demand instead of
# pre-reserving fixed-size blocks.
#
# Caveat: `expandable_segments:True` is **incompatible** with torch.compile's
# `cudagraph_trees` (PyTorch ≤2.8) — checkpoint pool state restore assumes the
# classic fixed-segment topology, and trips
# `RuntimeError: Expected curr_block->next == nullptr` during compiled warmup
# / replay. So when `--compile` is requested we skip the env override and let
# the default allocator run.
if "--compile" not in sys.argv:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import (
    closed_form_inverse_se3_general,
    unproject_depth_map_to_point_map,
)
from lingbot_map.utils.load_fn import load_and_preprocess_images


# =============================================================================
# Image loading
# =============================================================================

def load_images(image_folder=None, video_path=None, fps=10, image_ext=".jpg,.png,.JPG",
                first_k=None, stride=1, image_size=518, patch_size=14, num_workers=8,
                rotate_clockwise_90=False):
    """Load images from folder or video and preprocess into a tensor.

    Returns:
        (images, paths, resolved_image_folder): preprocessed tensor, file paths,
        and the folder containing the source images (for sky mask caching etc.).
    """
    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        paths = saved
        resolved_folder = out_dir
        print(f"Extracted {len(paths)} frames from video ({total_frames} total, interval={interval})")
    else:
        exts = image_ext.split(",")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        # A case-insensitive filesystem can return the same file for both
        # ``*.jpg`` and ``*.JPG``. Duplicate adjacent frames erase parallax and
        # noticeably degrade reconstruction, so dedupe by normalized path.
        paths = sorted({os.path.normcase(os.path.abspath(path)): path for path in paths}.values())
        resolved_folder = image_folder

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    if rotate_clockwise_90:
        rotated_dir = tempfile.mkdtemp(prefix="lingbot_rot_cw90_")
        rotated_paths = []
        # Image.ROTATE_270 = lossless 90° clockwise (270° counter-clockwise) reordering.
        for p in tqdm(paths, desc="Rotating images 90° CW"):
            out_path = os.path.join(rotated_dir, os.path.basename(p))
            Image.open(p).transpose(Image.ROTATE_270).save(out_path)
            rotated_paths.append(out_path)
        paths = rotated_paths
        resolved_folder = rotated_dir
        print(f"Rotated {len(paths)} images 90° clockwise → {rotated_dir}")

    print(f"Loading {len(paths)} images...")
    images = load_and_preprocess_images(
        paths,
        mode="crop",
        image_size=image_size,
        patch_size=patch_size,
    )
    h, w = images.shape[-2:]
    print(f"Preprocessed images to {w}x{h} using canonical crop mode")
    return images, paths, resolved_folder


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    """Load GCTStream model from checkpoint."""
    if getattr(args, "mode", "streaming") == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    print("Building model...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )

    if args.model_path:
        print(f"Loading checkpoint: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Checkpoint loaded.")

    return model.to(device).eval()


# =============================================================================
# torch.compile (opt-in via --compile)
# =============================================================================

def compile_model(model):
    """Compile hot, fixed-shape modules with mode="reduce-overhead".

    Mirrors the targets in gct_profile.py:compile_model. Unlike the profile script,
    `model.point_head` is **kept** — the demo needs world_points for visualization.
    """
    agg = model.aggregator
    for i, b in enumerate(agg.frame_blocks):
        agg.frame_blocks[i] = torch.compile(b, mode="reduce-overhead")
    for i, b in enumerate(agg.patch_embed.blocks):
        agg.patch_embed.blocks[i] = torch.compile(b, mode="reduce-overhead")
    for b in agg.global_blocks:
        if hasattr(b, 'attn_pre'):
            b.attn_pre = torch.compile(b.attn_pre, mode="reduce-overhead")
        if hasattr(b, 'ffn_residual'):
            b.ffn_residual = torch.compile(b.ffn_residual, mode="reduce-overhead")
        b.attn.proj = torch.compile(b.attn.proj, mode="reduce-overhead")


def _warm_streaming(model, images, scale_frames, warm_stream_n, dtype,
                    passes=1, keyframe_interval=1):
    """Drive `clean_kv_cache → Phase 1 → N streaming forwards` `passes` times.

    Warmup inputs are sliced from the already-preprocessed ``images`` tensor, so
    their **spatial shape (H×W) and number of scale frames adapt to the user's
    input** — this is what makes the captured CUDA graphs match what
    ``inference_streaming`` will replay (reduce-overhead mode keys on shape).

    The streaming loop alternates keyframe / non-keyframe forwards according to
    ``keyframe_interval``, mirroring ``inference_streaming``'s call pattern so
    the ``skip_append`` (defer+append+attend+rollback) path is also captured
    during warmup.  Without this, the first non-keyframe in the real run hits
    cold orchestration code and can confuse cudagraph_trees' allocator
    checkpoint state.
    """
    num_avail = int(images.shape[0])
    scale_frames = max(1, min(int(scale_frames), num_avail))
    # Keep at least one streaming frame for the per-frame compile path; if the
    # user supplied <= scale_frames images, shrink scale to free a stream slot.
    if scale_frames >= num_avail:
        scale_frames = max(1, num_avail - 1)
    warm_stream_n = max(1, min(int(warm_stream_n), num_avail - scale_frames))
    kf_int = max(int(keyframe_interval), 1)

    # images: [S, 3, H, W] on device already; slice + add batch dim, no copy of
    # spatial dims so warmup shape == real inference shape (H, W).
    warm_scale = images[:scale_frames].unsqueeze(0).to(dtype)
    warm_stream = images[scale_frames:scale_frames + warm_stream_n].unsqueeze(0).to(dtype)

    for _ in range(passes):
        model.clean_kv_cache()
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(
                warm_scale,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=scale_frames,
                causal_inference=True,
            )
        for i in range(warm_stream_n):
            is_keyframe = (kf_int <= 1) or (i % kf_int == 0)
            if not is_keyframe:
                model._set_skip_append(True)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                model.forward(
                    warm_stream[:, i:i + 1],
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
            if not is_keyframe:
                model._set_skip_append(False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Wipe warmup KV so real inference_streaming starts clean (it also calls
    # clean_kv_cache internally, but this is defensive + makes intent obvious).
    model.clean_kv_cache()


# =============================================================================
# Post-processing
# =============================================================================

_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}


def _squeeze_single_batch(key, value):
    """Drop the leading batch dimension for single-sequence demo outputs."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    # Convert w2c to c2w
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    print("Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def prepare_for_visualization(predictions, images=None):
    """Convert predictions to the unbatched NumPy format used by vis code."""
    vis_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = _squeeze_single_batch(k, v.detach().cpu())
            vis_predictions[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            vis_predictions[k] = _squeeze_single_batch(k, v)
        else:
            vis_predictions[k] = v

    if images is None:
        images = predictions.get("images")

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)
    elif isinstance(images, torch.Tensor):
        images = _squeeze_single_batch("images", images).numpy()

    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if images is not None:
        vis_predictions["images"] = images

    return vis_predictions


def validate_predictions(predictions, stage="inference"):
    """Reject numerically corrupt reconstructions before opening the viewer."""
    required = ("pose_enc", "depth", "depth_conf")
    problems = []

    for key in required:
        value = predictions.get(key)
        if not isinstance(value, torch.Tensor):
            problems.append(f"missing tensor {key!r}")
            continue

        finite = torch.isfinite(value)
        if finite.all():
            continue

        # Model outputs are [B, S, ...] before post-processing and [S, ...]
        # afterwards. Report the first affected frame in either representation.
        frame_dim = 1 if value.ndim == _BATCHED_NDIMS.get(key) else 0
        reduce_dims = tuple(i for i in range(value.ndim) if i != frame_dim)
        frame_ok = finite.all(dim=reduce_dims)
        bad_frames = (~frame_ok).nonzero(as_tuple=False).flatten().tolist()
        finite_pct = 100.0 * finite.float().mean().item()
        problems.append(
            f"{key} is {finite_pct:.2f}% finite; first bad frame={bad_frames[0]}"
        )

    depth = predictions.get("depth")
    if isinstance(depth, torch.Tensor) and torch.isfinite(depth).any():
        finite_depth = depth[torch.isfinite(depth)]
        positive_pct = 100.0 * (finite_depth > 0).float().mean().item()
        if positive_pct < 99.0:
            problems.append(f"depth is only {positive_pct:.2f}% positive")

    if problems:
        hint = ""
        if torch.version.hip is not None:
            hint = (
                " AMD/ROCm streaming SDPA is known to corrupt the growing KV cache; "
                "use --mode windowed --window_size 8 --overlap_size 4."
            )
        raise RuntimeError(
            f"Invalid reconstruction after {stage}: " + "; ".join(problems) + "." + hint
        )


def diagnose_geometry(predictions, conf_threshold=1.5, sample_step=8):
    """Print frame-level geometry anomalies without modifying predictions."""
    depth = predictions.get("depth")
    conf = predictions.get("depth_conf")
    extrinsic = predictions.get("extrinsic")
    intrinsic = predictions.get("intrinsic")
    if any(value is None for value in (depth, conf, extrinsic, intrinsic)):
        print("Geometry diagnostics skipped: depth/confidence/cameras unavailable")
        return []

    arrays = []
    for value in (depth, conf, extrinsic, intrinsic):
        arrays.append(value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value))
    depth, conf, extrinsic, intrinsic = arrays
    points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    centers = extrinsic[:, :3, 3]
    pose_steps = np.r_[0.0, np.linalg.norm(np.diff(centers, axis=0), axis=1)]

    rows = []
    for frame in range(len(points)):
        valid = np.isfinite(points[frame]).all(-1) & (conf[frame] > conf_threshold)
        cloud = points[frame][::sample_step, ::sample_step][valid[::sample_step, ::sample_step]]
        if len(cloud) < 16:
            rows.append((frame, np.nan, np.nan, pose_steps[frame], len(cloud)))
            continue
        centered = cloud - np.median(cloud, axis=0)
        covariance = centered.T @ centered / len(centered)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        planarity = float(eigenvalues[0] / max(eigenvalues.sum(), 1e-12))
        thickness = float(np.sqrt(eigenvalues[0]))
        rows.append((frame, planarity, thickness, pose_steps[frame], len(cloud)))

    finite_rows = [row for row in rows if np.isfinite(row[1])]
    median_step = np.median([row[3] for row in finite_rows[1:]]) if len(finite_rows) > 1 else 0.0
    ranked = sorted(
        finite_rows,
        key=lambda row: (row[1], -row[3]),
    )[:10]
    print("Geometry diagnostics (lowest PCA thickness ratio first):")
    print("  frame  planar_ratio  thickness  pose_step  points")
    for frame, planar, thickness, pose_step, count in ranked:
        flags = []
        if planar < 1e-4:
            flags.append("SLAB")
        if median_step > 0 and pose_step > 3 * median_step:
            flags.append("POSE_JUMP")
        print(
            f"  {frame:5d}  {planar:12.6g}  {thickness:9.4g}  "
            f"{pose_step:9.4g}  {count:6d}  {' '.join(flags)}"
        )
    print("Largest camera-pose steps:")
    for frame, planar, thickness, pose_step, count in sorted(
        finite_rows[1:], key=lambda row: row[3], reverse=True
    )[:8]:
        ratio = pose_step / median_step if median_step > 0 else np.nan
        boundary = "WINDOW_EDGE" if frame % 4 == 0 else ""
        print(f"  frame {frame:3d}: step={pose_step:.5g} ({ratio:.2f}x median) {boundary}")

    chunk_scales = predictions.get("chunk_scales")
    if chunk_scales is not None:
        chunk_scales = (
            chunk_scales.detach().cpu().numpy()
            if torch.is_tensor(chunk_scales) else np.asarray(chunk_scales)
        )
        print(f"Window alignment scales: {np.array2string(chunk_scales.squeeze(), precision=4)}")
    return rows


def parse_frame_spec(spec):
    """Parse comma-separated frame numbers and inclusive ranges."""
    frames = set()
    if not spec:
        return frames
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid frame range: {token!r}")
            frames.update(range(start, end + 1))
        else:
            frame = int(token)
            if frame < 0:
                raise ValueError(f"invalid frame number: {token!r}")
            frames.add(frame)
    return frames


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP: Streaming 3D Reconstruction Demo")

    # Input
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rotate_clockwise_90", action="store_true",
                        help="Rotate source images 90° clockwise before preprocessing "
                             "(crop/resize then operates on the rotated aspect ratio)")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)

    # Inference mode
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "windowed"],
                        help="streaming: frame-by-frame with KV cache; windowed: overlapping windows for long sequences")

    # Streaming options
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=None,
        help="Every N-th frame after scale frames is kept as a keyframe. 1 = every frame. "
            "Streaming: if unset, auto-selected (1 when num_frames <= 320, else ceil(num_frames / 320)) "
            "to bound KV cache. Windowed: defaults to 1; --window_size counts keyframes, so values >1 "
            "expand each window's actual-frame coverage to "
            "scale_frames + (window_size - scale_frames) * keyframe_interval.",
    )
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4,
                        help="Camera head iterative-refinement steps. Default 4; set 1 for faster inference "
                            "(skips 3 refinement passes at a small accuracy cost).")
    parser.add_argument("--use_sdpa", action="store_true", default=False,
                        help="Use SDPA backend (no flashinfer needed). Default: FlashInfer")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="torch.compile hot modules (reduce-overhead) with a CUDA-graph warmup. "
                            "Streaming mode only; ~5 FPS faster at 518x378. Adds ~30-60 s warmup time.")
    parser.add_argument(
        "--offload_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload per-frame predictions to CPU during inference to cut GPU peak memory "
            "(on by default).  Use --no-offload_to_cpu to keep outputs on GPU.",
    )
    # Windowed options
    parser.add_argument("--window_size", type=int, default=64, help="Frames per window (windowed mode)")
    parser.add_argument("--overlap_size", type=int, default=16,
                        help="Overlap between windows in *actual frames*")
    parser.add_argument("--overlap_keyframes", type=int, default=None,
                        help="Overlap expressed in *keyframes* (takes precedence over "
                             "--overlap_size). Converted internally to "
                             "max(num_scale_frames, overlap_keyframes * keyframe_interval) "
                             "actual frames.  Recommended when --keyframe_interval > 1.")

    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument(
        "--depth_edge_threshold", type=float, default=None,
        help="Remove points beside relative depth jumps larger than this ratio; "
             "0 disables it (AMD inspection preset: 0.15).",
    )
    parser.add_argument(
        "--show_camera", action=argparse.BooleanOptionalAction, default=None,
        help="Show camera frustums in the viewer (AMD inspection preset: off).",
    )
    parser.add_argument(
        "--allow_unsafe_rocm_streaming", action="store_true", default=False,
        help="Allow known-corrupt growing SDPA KV-cache paths on ROCm.",
    )
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument("--sky_mask_dir", type=str, default=None,
                        help="Directory for cached sky masks (default: <image_folder>_sky_masks/)")
    parser.add_argument("--sky_mask_visualization_dir", type=str, default=None,
                        help="Save sky mask visualizations (original | mask | overlay) to this directory")
    parser.add_argument("--export_preprocessed", type=str, default=None,
                        help="Export stride-sampled, resized/cropped images to this folder")
    parser.add_argument("--diagnose_geometry", action="store_true",
                        help="Print frame-level slab and pose-jump diagnostics")
    parser.add_argument(
        "--exclude_point_frames", type=str, default="",
        help="Exclude frames from point-cloud display, but keep them in inference. "
             "Accepts comma-separated indices/ranges such as '3,8-10'.",
    )

    args = parser.parse_args()
    assert args.image_folder or args.video_path, \
        "Provide --image_folder or --video_path"
    try:
        args.exclude_point_frames = parse_frame_spec(args.exclude_point_frames)
    except ValueError as error:
        parser.error(str(error))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The CUDA-shaped PyTorch API is also used by ROCm. On AMD, FlashInfer is
    # unavailable and the growing dict-based SDPA cache currently becomes
    # non-finite after the first streamed frame. Short overlapping windows keep
    # each window entirely in the stable scale phase.
    is_rocm = torch.version.hip is not None
    if is_rocm:
        args.use_sdpa = True
        mode_was_explicit = "--mode" in sys.argv
        if args.mode == "streaming":
            if mode_was_explicit and not args.allow_unsafe_rocm_streaming:
                parser.error(
                    "ROCm streaming SDPA produces non-finite predictions. "
                    "Use --mode windowed (recommended), or explicitly pass "
                    "--allow_unsafe_rocm_streaming for diagnostics."
                )
            if not mode_was_explicit:
                args.mode = "windowed"
        if args.mode == "windowed":
            if "--window_size" not in sys.argv:
                args.window_size = 8
            if "--overlap_size" not in sys.argv and "--overlap_keyframes" not in sys.argv:
                args.overlap_size = 4
            if (
                args.window_size > args.num_scale_frames
                and not args.allow_unsafe_rocm_streaming
            ):
                parser.error(
                    "ROCm SDPA becomes non-finite when a window grows beyond "
                    f"the {args.num_scale_frames} scale frames. Use "
                    f"--window_size {args.num_scale_frames}, or explicitly pass "
                    "--allow_unsafe_rocm_streaming for diagnostics."
                )
        if "--conf_threshold" not in sys.argv:
            args.conf_threshold = 1.5
        if "--downsample_factor" not in sys.argv:
            args.downsample_factor = 2
        if "--point_size" not in sys.argv:
            args.point_size = 0.001
        if args.depth_edge_threshold is None:
            args.depth_edge_threshold = 0.15
        if args.show_camera is None:
            args.show_camera = False
        print(
            "ROCm preset: SDPA + "
            f"{args.mode} mode"
            + (f" (window={args.window_size}, overlap={args.overlap_size})" if args.mode == "windowed" else "")
        )
    elif args.show_camera is None:
        args.show_camera = True
    if args.depth_edge_threshold is None:
        args.depth_edge_threshold = 0.0

    # ── Load images & model ──────────────────────────────────────────────────
    t0 = time.time()
    images, paths, resolved_image_folder = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
        rotate_clockwise_90=args.rotate_clockwise_90,
    )

    # Export preprocessed images if requested
    if args.export_preprocessed:
        os.makedirs(args.export_preprocessed, exist_ok=True)
        print(f"Exporting {images.shape[0]} preprocessed images to {args.export_preprocessed}...")
        for i in range(images.shape[0]):
            img = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(args.export_preprocessed, f"{i:06d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        print(f"Exported to {args.export_preprocessed}")

    model = load_model(args, device)
    print(f"Total load time: {time.time() - t0:.1f}s")

    # Pick inference dtype; autocast still runs for the ops that need fp32 (e.g. LayerNorm).
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # Cast the aggregator (DINOv2-style trunk) to the inference dtype to remove the
    # redundant fp32 master weight copy + autocast bf16 weight cache (~2-3 GB saved,
    # no measurable quality change). gct_base._predict_* upcasts inputs to fp32 and
    # runs each head under `autocast(enabled=False)`, so camera/depth/point heads
    # keep fp32 weights automatically.
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype} (heads kept in fp32)")
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)
    num_frames = images.shape[0]
    print(f"Input: {num_frames} frames, shape {tuple(images.shape)}")
    print(f"Mode: {args.mode}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU mem after load: "
            f"alloc={torch.cuda.memory_allocated()/1e9:.2f} GB, "
            f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB"
        )

    if args.keyframe_interval is None:
        if args.mode == "streaming" and num_frames > 320:
            args.keyframe_interval = (num_frames + 319) // 320
            print(
                f"Auto-selected --keyframe_interval={args.keyframe_interval} "
                f"(num_frames={num_frames} > 320)."
            )
        else:
            args.keyframe_interval = 1

    if args.keyframe_interval > 1:
        if args.mode == "streaming":
            print(
                f"Keyframe streaming enabled: interval={args.keyframe_interval} "
                f"(after the first {args.num_scale_frames} scale frames)."
            )
        else:  # windowed
            actual_per_window = (
                args.num_scale_frames
                + max(0, args.window_size - args.num_scale_frames) * args.keyframe_interval
            )
            print(
                f"Keyframe windowed enabled: interval={args.keyframe_interval}, "
                f"each window covers up to {actual_per_window} actual frames "
                f"(window_size={args.window_size} keyframes, scale={args.num_scale_frames})."
            )

    # ── Optional: torch.compile + CUDA-graph warmup (streaming only) ────────
    if args.compile:
        if args.mode != "streaming":
            print(
                f"--compile only applies to --mode streaming (got {args.mode!r}); "
                "skipping compile."
            )
        else:
            scale_for_warm = min(args.num_scale_frames, num_frames)
            if scale_for_warm >= num_frames:
                scale_for_warm = max(1, num_frames - 1)
            warm_stream_n = min(10, max(1, num_frames - scale_for_warm))
            warm_h, warm_w = int(images.shape[-2]), int(images.shape[-1])
            print(
                f"Warmup eager (scale={scale_for_warm} + {warm_stream_n} streaming, "
                f"shape={warm_h}x{warm_w}, kf_int={args.keyframe_interval})..."
            )
            t_warm = time.time()
            _warm_streaming(
                model, images, scale_for_warm, warm_stream_n, dtype,
                passes=1, keyframe_interval=args.keyframe_interval,
            )
            print(f"  eager warmup: {time.time() - t_warm:.1f}s")

            print("Compiling hot modules...")
            compile_model(model)

            # 3 passes under compile: 1st captures CUDA graphs, 2nd/3rd replay so
            # the caching allocator / graph-address map converge on the state the
            # real inference will see. See gct_profile.py:302-306 for rationale.
            print("Warmup compiled (3x dress rehearsal)...")
            t_warm = time.time()
            _warm_streaming(
                model, images, scale_for_warm, warm_stream_n, dtype,
                passes=3, keyframe_interval=args.keyframe_interval,
            )
            print(f"  compiled warmup: {time.time() - t_warm:.1f}s")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"Running {args.mode} inference (dtype={dtype})...")
    t0 = time.time()

    output_device = torch.device("cpu") if args.offload_to_cpu else None

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )
        else:  # windowed
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                overlap_keyframes=args.overlap_keyframes,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device
            )

    print(f"Inference done in {time.time() - t0:.1f}s")
    validate_predictions(predictions)
    if torch.cuda.is_available():
        print(
            f"GPU peak during inference: "
            f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB "
            f"(reserved peak {torch.cuda.max_memory_reserved()/1e9:.2f} GB)"
        )

    # ── Post-process ─────────────────────────────────────────────────────────
    if args.offload_to_cpu:
        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        images_for_post = predictions["images"]  # already CPU
    else:
        images_for_post = images

    predictions, images_cpu = postprocess(predictions, images_for_post)
    if args.diagnose_geometry:
        diagnose_geometry(predictions, conf_threshold=args.conf_threshold)

    # ── Visualize ────────────────────────────────────────────────────────────
    try:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=prepare_for_visualization(predictions, images_cpu),
            port=args.port,
            vis_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            point_size=args.point_size,
            show_camera=args.show_camera,
            mask_sky=args.mask_sky,
            image_folder=resolved_image_folder,
            sky_mask_dir=args.sky_mask_dir,
            sky_mask_visualization_dir=args.sky_mask_visualization_dir,
            depth_edge_threshold=args.depth_edge_threshold,
            excluded_frames=args.exclude_point_frames,
        )
        print(f"3D viewer at http://localhost:{args.port}")
        viewer.run()
    except ImportError:
        print("viser not installed. Install with: pip install lingbot-map[vis]")
        print(f"Predictions contain keys: {list(predictions.keys())}")


if __name__ == "__main__":
    main()
