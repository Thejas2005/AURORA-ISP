
"""Real-data evaluation of Traditional / Hybrid / Learned on held-out SIDD-CC captures.

Run from project root:
python -m experiments.benchmarks.evaluate_real --data-root "PATH_TO_SIDD_CC" --checkpoint models/checkpoints/aurora_v4_best.pt --crop 1024 --max-captures 4 --out eval_smoke

Conditioning policies:
  oracle    : ground-truth-derived conditioning; diagnostic only
  estimated : conditioning estimated by the current inference pipeline
  neutral   : noise_sigma=0, mean_luma=0.5

Oracle conditioning is NOT available during deployment.
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from core.pipeline import traditional_isp, hybrid_isp, learned_isp
from core.evaluation.metrics import all_metrics

POLICIES = ("oracle", "estimated", "neutral")
METRICS = ("PSNR", "SSIM", "DeltaE2000", "LPIPS")


def centre_crop(size, *arrays):
    h, w = arrays[0].shape[:2]

    if any(a.shape[:2] != (h, w) for a in arrays):
        raise ValueError("RAW and target image dimensions do not match.")

    if size == 0 or size >= min(h, w):
        return arrays

    size -= size % 2

    if size < 8:
        raise ValueError("Crop size must be 0 or at least 8.")

    y0 = ((h - size) // 2) // 2 * 2
    x0 = ((w - size) // 2) // 2 * 2

    return tuple(
        a[y0:y0 + size, x0:x0 + size]
        for a in arrays
    )


def meta_for(base, policy, noisy, gt_raw, target):
    """Construct conditioning metadata for one evaluation policy."""

    m = dict(base)

    if policy == "oracle":
        # Ground-truth-derived values used by the v4.0.2 training loader.
        m["noise_sigma"] = float(np.std(noisy - gt_raw))
        m["mean_luma"] = float(np.clip(target.mean(), 0, 1))

    elif policy == "estimated":
        # Remove placeholders so the inference pipeline estimates them.
        m.pop("noise_sigma", None)
        m.pop("mean_luma", None)

    elif policy == "neutral":
        m["noise_sigma"] = 0.0
        m["mean_luma"] = 0.5

    else:
        raise ValueError(f"Unknown conditioning policy: {policy}")

    return m


def run_capture(noisy, gt_raw, target, base_meta, learned=None,
                warmup=1, runs=3):
    """Evaluate every mode on the same capture."""

    jobs = [
        ("traditional",
         lambda: traditional_isp(noisy, base_meta)),

        ("hybrid_classical",
         lambda: hybrid_isp(noisy, base_meta)[0]),
    ]

    if learned is not None:

        for policy in POLICIES:
            mp = meta_for(
                base_meta, policy, noisy, gt_raw, target
            )

            jobs.append((
                f"learned_{policy}",
                lambda mp=mp: learned_isp(noisy, mp, learned)
            ))

        estimated_meta = meta_for(
            base_meta, "estimated", noisy, gt_raw, target
        )

        jobs.append((
            "hybrid_learned_estimated",
            lambda: hybrid_isp(
                noisy, estimated_meta, learned
            )[0]
        ))

    results = {}

    for name, fn in jobs:

        for _ in range(warmup):
            fn()

        timings = []
        prediction = None

        for _ in range(runs):
            t0 = time.perf_counter()
            prediction = fn()
            elapsed = (time.perf_counter() - t0) * 1000
            timings.append(elapsed)

        metrics = all_metrics(prediction, target)

        results[name] = {
            **metrics,
            "latency_ms": float(statistics.median(timings)),
            "latency_ms_mean": float(statistics.mean(timings)),
        }

    return results


def summarise(rows):
    modes = list(dict.fromkeys(
        r["mode"] for r in rows
    ))

    summary = {}

    for mode in modes:

        mode_rows = [
            r for r in rows if r["mode"] == mode
        ]

        item = {
            "n_captures": len(mode_rows),
            "latency_ms_median": float(
                statistics.median(
                    r["latency_ms"] for r in mode_rows
                )
            )
        }

        for metric in METRICS:

            values = [
                float(r[metric])
                for r in mode_rows
                if r.get(metric) is not None
            ]

            item[metric] = (
                {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
                if values else None
            )

        summary[mode] = item

    traditional = {
        r["capture"]: r
        for r in rows
        if r["mode"] == "traditional"
    }

    for mode in modes:

        if mode == "traditional":
            continue

        deltas = [
            r["PSNR"] - traditional[r["capture"]]["PSNR"]
            for r in rows
            if r["mode"] == mode
            and r["capture"] in traditional
        ]

        if deltas:
            summary[mode]["PSNR_minus_traditional"] = {
                "mean_db": float(np.mean(deltas)),
                "median_db": float(np.median(deltas)),
                "wins": int(sum(x > 0 for x in deltas)),
                "of": len(deltas)
            }

    return summary


def load_learned(path):
    import torch

    from models.learned_isp.unet import AuroraUNet
    from models.learned_isp.adapter import LearnedISP

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if "model_state" not in payload:
        raise ValueError(
            "Checkpoint does not contain model_state."
        )

    cfg = payload.get("model_config", {})

    model = AuroraUNet(
        base=cfg.get("base", 24),
        meta_dim=cfg.get("meta_dim", 5),
        use_meta=cfg.get("use_meta", True)
    )

    model.load_state_dict(
        payload["model_state"],
        strict=True
    )

    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    learned = LearnedISP(model, device=device)

    info = {
        "epoch": payload.get("epoch"),
        "val_l1": payload.get("val_l1"),
        "val_psnr_db": payload.get("val_psnr_db"),
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "checkpoint_MB": path.stat().st_size / 1e6,
        "device": device,
        "torch_version": torch.__version__
    }

    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)

    return learned, info


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", default=None)

    ap.add_argument(
        "--crop",
        type=int,
        default=1024,
        help="Centre crop in sensor pixels; 0 = full frame"
    )

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1
    )

    ap.add_argument("--max-captures", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="eval_out")

    args = ap.parse_args()

    if args.crop < 0:
        ap.error("--crop cannot be negative")

    if args.runs < 1 or args.warmup < 0:
        ap.error("--runs must be >= 1 and --warmup >= 0")

    if args.max_captures < 0:
        ap.error("--max-captures cannot be negative")

    from PIL import Image

    from datasets.loader import (
        PairedRawDataset,
        _mat_value,
        _read_metadata
    )

    dataset = PairedRawDataset(
        args.data_root,
        "val",
        32,
        args.validation_fraction,
        args.seed,
        1
    )

    learned, checkpoint_info = (
        load_learned(args.checkpoint)
        if args.checkpoint
        else (None, {})
    )

    records = dataset.records

    if args.max_captures:
        records = records[:args.max_captures]

    if not records:
        raise RuntimeError(
            "No validation captures were found."
        )

    rows = []

    for i, record in enumerate(records, 1):

        inst, npath, gpath, mpath, rpath = record

        noisy = np.asarray(
            _mat_value(npath, "data"),
            dtype=np.float32
        ).squeeze()

        gt_raw = np.asarray(
            _mat_value(gpath, "data"),
            dtype=np.float32
        ).squeeze()

        with Image.open(rpath) as image:
            target = (
                np.asarray(
                    image.convert("RGB"),
                    dtype=np.float32
                ) / 255.0
            )

        noisy, gt_raw, target = centre_crop(
            args.crop,
            noisy,
            gt_raw,
            target
        )

        base_meta = _read_metadata(mpath, inst)

        results = run_capture(
            noisy,
            gt_raw,
            target,
            base_meta,
            learned,
            args.warmup,
            args.runs
        )

        for mode, metrics in results.items():

            rows.append({
                "capture": npath.stem,
                "instance": inst,
                "mode": mode,
                **metrics
            })

        line = " | ".join(
            f"{mode}={value['PSNR']:.2f} dB"
            for mode, value in results.items()
        )

        print(
            f"[{i}/{len(records)}] {npath.stem}: {line}",
            flush=True
        )

    summary = summarise(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metadata = {
        "split": "validation, instance-level",
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "n_captures": len(records),
        "crop": args.crop,
        "reference": "camera-rendered clean sRGB PNG",
        "checkpoint": checkpoint_info,
        "warmup": args.warmup,
        "timed_runs": args.runs,
        "conditioning_note": (
            "Oracle uses GT RAW and target sRGB; not deployable. "
            "Estimated uses inference image analysis. "
            "Neutral uses noise_sigma=0 and mean_luma=0.5."
        )
    }

    (out / "summary.json").write_text(
        json.dumps({
            "meta": metadata,
            "summary": summary
        }, indent=2),
        encoding="utf-8"
    )

    with (out / "per_capture.csv").open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nFINAL RESULTS")
    print(
        f"{'MODE':30s} {'PSNR':>8s} "
        f"{'SSIM':>8s} {'dE2000':>9s} "
        f"{'LPIPS':>8s} {'ms':>9s}"
    )

    for mode, values in summary.items():

        def fmt(metric):
            item = values.get(metric)
            return (
                f"{item['mean']:.3f}"
                if item else "n/a"
            )

        print(
            f"{mode:30s} "
            f"{fmt('PSNR'):>8s} "
            f"{fmt('SSIM'):>8s} "
            f"{fmt('DeltaE2000'):>9s} "
            f"{fmt('LPIPS'):>8s} "
            f"{values['latency_ms_median']:9.1f}"
        )

    print(f"\nSaved: {out / 'summary.json'}")
    print(f"Saved: {out / 'per_capture.csv'}")


if __name__ == "__main__":
    main()
