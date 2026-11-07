
import sys
from pathlib import Path

# Add AURORA project root to Python's module search path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import time

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from core.denoising.classical import denoise


def load_rgb(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")

    return np.asarray(
        Image.open(path).convert("RGB"),
        dtype=np.uint8
    )


def calculate_metrics(pred, reference):
    pred_f = pred.astype(np.float32) / 255.0
    ref_f = reference.astype(np.float32) / 255.0

    return {
        "PSNR": float(
            peak_signal_noise_ratio(
                ref_f, pred_f, data_range=1.0
            )
        ),
        "SSIM": float(
            structural_similarity(
                ref_f,
                pred_f,
                channel_axis=-1,
                data_range=1.0
            )
        ),
        "DeltaE2000": float(
            np.mean(
                deltaE_ciede2000(
                    rgb2lab(pred_f),
                    rgb2lab(ref_f)
                )
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="AURORA RGB denoising evaluation"
    )

    parser.add_argument(
        "--noisy",
        default="test_images/noisy_parrots.png"
    )

    parser.add_argument(
        "--reference",
        default="test_images/clean_parrots.jpg"
    )

    parser.add_argument(
        "--output",
        default="outputs/parrot_test"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AURORA ISP - RGB DENOISING EVALUATION")
    print("=" * 60)

    print("Project root:", PROJECT_ROOT)

    noisy = load_rgb(args.noisy)
    reference = load_rgb(args.reference)

    if noisy.shape != reference.shape:
        raise ValueError(
            f"Image dimensions differ: "
            f"{noisy.shape} vs {reference.shape}"
        )

    print("Input dimensions:", noisy.shape)
    print("Device: CPU (classical OpenCV denoising)")
    print()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Baseline: original noisy image
    baseline_metrics = calculate_metrics(noisy, reference)

    results.append({
        "method": "noisy_input",
        **baseline_metrics,
        "latency_ms": 0.0
    })

    print(
        f"Noisy input: "
        f"PSNR={baseline_metrics['PSNR']:.3f} | "
        f"SSIM={baseline_metrics['SSIM']:.4f} | "
        f"DeltaE={baseline_metrics['DeltaE2000']:.3f}"
    )

    # Evaluate AURORA's actual classical RGB denoiser
    strengths = [0.3, 0.5, 0.7, 1.0]

    for strength in strengths:

        # Warm-up
        denoise(
            noisy.astype(np.float32) / 255.0,
            strength
        )

        # Repeated timing
        timings = []

        for _ in range(3):
            start = time.perf_counter()

            denoised = denoise(
                noisy.astype(np.float32) / 255.0,
                strength
            )

            timings.append(
                (time.perf_counter() - start) * 1000
            )

        latency = float(np.median(timings))

        # Convert output to uint8 RGB
        denoised = np.asarray(denoised)

        if denoised.dtype != np.uint8:
            if denoised.max() <= 1.0:
                denoised = denoised * 255.0

            denoised = np.clip(
                denoised, 0, 255
            ).astype(np.uint8)

        metrics = calculate_metrics(
            denoised, reference
        )

        name = f"aurora_nlm_{strength:.1f}"

        Image.fromarray(denoised).save(
            output_dir / f"{name}.png"
        )

        results.append({
            "method": name,
            **metrics,
            "latency_ms": latency
        })

        print(
            f"{name}: "
            f"PSNR={metrics['PSNR']:.3f} | "
            f"SSIM={metrics['SSIM']:.4f} | "
            f"DeltaE={metrics['DeltaE2000']:.3f} | "
            f"Latency={latency:.2f} ms"
        )

    # Save metrics
    csv_path = output_dir / "metrics.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print("Results directory:", output_dir.resolve())
    print("Metrics CSV:", csv_path.resolve())


if __name__ == "__main__":
    main()
