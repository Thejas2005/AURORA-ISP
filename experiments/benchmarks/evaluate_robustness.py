"""Controlled RAW-domain robustness benchmark for AURORA.

Perturbations are applied to the noisy Bayer mosaic only; clean RAW and camera
rendered sRGB references remain unchanged. This is a stress test, not a claim
that synthetic perturbations exactly model every sensor.
"""
from __future__ import annotations
import argparse, csv, json, statistics, time
from pathlib import Path
import numpy as np
from PIL import Image

from experiments.benchmarks.evaluate_real import (
    load_learned, centre_crop, meta_for, summarise, METRICS
)
from datasets.loader import PairedRawDataset, _mat_value, _read_metadata
from core.pipeline import traditional_isp, learned_isp, hybrid_isp
from core.evaluation.metrics import all_metrics
from core.raw.bayer import PATTERNS


def sync(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def make_scenarios(noisy, clean_raw, meta, seed=2027):
    """Return scenario name, perturbed RAW, metadata and scenario parameters."""
    rng = np.random.default_rng(seed)
    residual = noisy.astype(np.float32) - clean_raw.astype(np.float32)
    # Controlled noise scale estimated from paired RAW solely for this benchmark.
    base_sigma = float(np.std(residual))
    scenarios = [("baseline", noisy.copy(), dict(meta), {"type": "none"})]

    for label, multiplier in (("noise_mild", 1.25), ("noise_moderate", 1.75), ("noise_severe", 2.5)):
        extra_sigma = base_sigma * np.sqrt(max(multiplier * multiplier - 1.0, 0.0))
        perturbed = noisy + rng.normal(0.0, extra_sigma, noisy.shape).astype(np.float32)
        scenarios.append((label, perturbed.astype(np.float32), dict(meta),
                          {"type": "additive_gaussian_raw", "base_sigma": base_sigma,
                           "extra_sigma": float(extra_sigma), "total_target_sigma": float(base_sigma*multiplier)}))

    # Exposure changes on the sensor mosaic. Values are clipped to the normalized
    # sensor interval; clipping is separately reported because lost highlights
    # cannot be recovered reliably.
    for label, factor in (("underexposure_2x", 0.5), ("underexposure_4x", 0.25),
                          ("overexposure_1_5x", 1.5), ("overexposure_2x", 2.0)):
        perturbed = np.minimum(noisy * factor, 1.0).astype(np.float32)
        m = dict(meta)
        m["exposure_ms"] = max(float(m.get("exposure_ms", 1.0)) * factor, 1e-4)
        scenarios.append((label, perturbed, m,
                          {"type": "exposure_scale", "factor": factor,
                           "clipped_fraction": float(np.mean((noisy*factor) >= 1.0))}))

    # Inject a modest sensor channel-gain / white-balance error in CFA planes.
    # This tests generalisation: current learned metadata contract has no explicit
    # WB gains, so it is not supplied a hidden correction unavailable to the model.
    pattern = str(meta.get("pattern", "RGGB")).upper()
    for label, gains in (("wb_warm_cast", (1.15, 1.0, 0.82)),
                         ("wb_cool_cast", (0.82, 1.0, 1.18))):
        planes = {
            "RGGB": (gains[0], gains[1], gains[1], gains[2]),
            "BGGR": (gains[2], gains[1], gains[1], gains[0]),
            "GRBG": (gains[1], gains[0], gains[2], gains[1]),
            "GBRG": (gains[1], gains[2], gains[0], gains[1]),
        }[pattern]
        perturbed = noisy.copy().astype(np.float32)
        for (yy, xx), gain in zip(((0,0),(0,1),(1,0),(1,1)), planes):
            perturbed[yy::2, xx::2] *= gain
        scenarios.append((label, np.clip(perturbed, 0.0, 1.0), dict(meta),
                          {"type": "cfa_channel_gain", "rgb_gains": gains,
                           "note": "No WB correction metadata supplied to learned model"}))

    # Hot-pixel stress: sparse positive impulses at deterministic locations.
    hot = noisy.copy().astype(np.float32)
    count = max(1, int(hot.size * 0.0005))
    yy = rng.integers(0, hot.shape[0], count)
    xx = rng.integers(0, hot.shape[1], count)
    hot[yy, xx] = 1.0
    scenarios.append(("hot_pixels", hot, dict(meta),
                      {"type": "sparse_hot_pixels", "fraction": count/hot.size}))
    return scenarios


def save_rgb(image, path):
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError("Invalid RGB output")
    Image.fromarray(np.round(np.clip(x, 0, 1)*255).astype(np.uint8), "RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", default="models/checkpoints/aurora_v4_best.pt")
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--max-captures", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--out", default="outputs/raw_robustness")
    args = ap.parse_args()
    if args.max_captures < 1 or args.runs < 1 or args.warmup < 0:
        ap.error("max-captures and runs must be >=1; warmup >=0")

    dataset = PairedRawDataset(args.data_root, "val", 32, args.validation_fraction, args.seed, 1)
    records = dataset.records if args.max_captures == 0 else dataset.records[:args.max_captures]
    learned, ckpt = load_learned(args.checkpoint)
    device = str(learned.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, scenario_specs = [], {}
    if device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()

    for ci, record in enumerate(records, 1):
        instance, noisy_path, gt_path, meta_path, rgb_path = record
        noisy = np.asarray(_mat_value(noisy_path, "data"), dtype=np.float32).squeeze()
        clean_raw = np.asarray(_mat_value(gt_path, "data"), dtype=np.float32).squeeze()
        with Image.open(rgb_path) as im:
            target = np.asarray(im.convert("RGB"), dtype=np.float32)/255.0
        noisy, clean_raw, target = centre_crop(args.crop, noisy, clean_raw, target)
        base_meta = _read_metadata(meta_path, instance)
        variants = make_scenarios(noisy, clean_raw, base_meta, args.seed + ci)
        for si, (scenario, raw_variant, meta, spec) in enumerate(variants):
            scenario_specs.setdefault(scenario, {})[noisy_path.stem] = spec
            jobs = {
                "traditional": lambda raw_variant=raw_variant, meta=meta: traditional_isp(raw_variant, meta),
                "learned_estimated": lambda raw_variant=raw_variant, meta=meta: learned_isp(
                    raw_variant, meta_for(meta, "estimated", raw_variant, clean_raw, target), learned),
                "hybrid_estimated": lambda raw_variant=raw_variant, meta=meta: hybrid_isp(
                    raw_variant, meta_for(meta, "estimated", raw_variant, clean_raw, target), learned)[0],
            }
            for mode, fn in jobs.items():
                pred = None
                for _ in range(args.warmup):
                    _ = fn()
                times = []
                for _ in range(args.runs):
                    sync(device); t0 = time.perf_counter()
                    pred = fn()
                    sync(device); times.append((time.perf_counter()-t0)*1000)
                met = all_metrics(pred, target)
                row = {"capture": noisy_path.stem, "instance": instance, "scenario": scenario,
                       "mode": mode, **met, "latency_ms": float(statistics.median(times))}
                rows.append(row)
                if args.save_images:
                    folder = out / "images" / scenario / noisy_path.stem
                    folder.mkdir(parents=True, exist_ok=True)
                    save_rgb(pred, folder / f"{mode}.png")
            print(f"[{ci}/{len(records)}] {noisy_path.stem} / {scenario}: " +
                  ", ".join(f"{r['mode']}={r['PSNR']:.2f}dB" for r in rows[-3:]), flush=True)

    # Scenario x mode aggregate with mean/std, plus per-capture rows.
    summary = {}
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        summary[scenario] = {}
        for mode in dict.fromkeys(r["mode"] for r in rows):
            group = [r for r in rows if r["scenario"] == scenario and r["mode"] == mode]
            if not group: continue
            summary[scenario][mode] = {
                metric: {"mean": float(np.mean([r[metric] for r in group if r[metric] is not None])),
                         "std": float(np.std([r[metric] for r in group if r[metric] is not None]))}
                if any(r[metric] is not None for r in group) else None
                for metric in METRICS
            }
            summary[scenario][mode]["latency_ms_median"] = float(np.median([r["latency_ms"] for r in group]))
    report = {"meta": {"dataset": "SIDD-CC validation instance split", "captures": len(records),
                       "seed": args.seed, "checkpoint": ckpt, "device": device,
                       "cuda_synchronized_timing": device == "cuda",
                       "scenario_parameters": scenario_specs,
                       "warning": "Synthetic stress tests; not proof of all real-world camera scenarios."},
              "summary": summary}
    (out/"robustness_summary.json").write_text(json.dumps(report, indent=2))
    with (out/"per_capture_scenario.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if device == "cuda":
        report["meta"]["peak_cuda_memory_mb"] = float(torch.cuda.max_memory_allocated()/1024**2)
        (out/"robustness_summary.json").write_text(json.dumps(report, indent=2))
    print("\nSaved:", (out/"robustness_summary.json").resolve())
    print("Saved:", (out/"per_capture_scenario.csv").resolve())
    if args.save_images: print("Images:", (out/"images").resolve())


if __name__ == "__main__":
    main()
