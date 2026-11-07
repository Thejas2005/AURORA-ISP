"""SIDD-CC paired Bayer RAW -> camera-rendered sRGB patch dataset.

Pairs are discovered by exact capture stem; train/validation split is by instance
(directory), before patch sampling. MAT RAW values are used as stored: SIDD-CC
RAW is already black-level-subtracted/normalised and may contain negatives.
"""
from __future__ import annotations
import re, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image
from core.raw.bayer import pack_bayer
from core.metadata import metadata_vector


def _mat_value(path: Path, key: str):
    data = loadmat(path, variable_names=[key])
    if key not in data:
        raise KeyError(f"{key!r} missing from {path}")
    return data[key]


def _scalar(meta, key, default):
    value = meta.get(key)
    if value is None:
        return default
    arr = np.asarray(value).squeeze()
    if arr.size != 1:
        return default
    try:
        return float(arr)
    except (TypeError, ValueError):
        return default


def _cfa(meta):
    value = meta.get("cfa_str")
    if value is None:
        raise ValueError("Metadata MAT has no cfa_str; refusing to guess Bayer order")
    arr = np.asarray(value).squeeze()
    if arr.dtype.kind in "US":
        text = "".join(str(x) for x in arr.reshape(-1)).strip()
    else:
        try:
            text = "".join(chr(int(x)) for x in arr.reshape(-1) if int(x)).strip()
        except Exception as exc:
            raise ValueError("Cannot decode cfa_str") from exc
    text = text.upper()
    if text not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
        raise ValueError(f"Unsupported cfa_str {text!r}")
    return text


def _read_metadata(path: Path, instance_name: str):
    raw = loadmat(path)
    cfa = _cfa(raw)
    # Capture name convention: <id>_<scene>_<camera>_<ISO>_<shutter>_...
    tokens = instance_name.split("_")
    iso = 100.0
    if len(tokens) >= 4 and tokens[3].isdigit():
        iso = float(tokens[3])
    # Exposure units in the directory token are not assumed. Keep neutral default.
    meta = {
        "pattern": cfa, "black_level": 0.0, "white_level": 1.0,
        "iso": iso, "exposure_ms": 1.0, "analog_gain": 1.0,
        "noise_sigma": 0.0, "mean_luma": 0.5,
        "metadata_provenance": {"iso": "parsed_from_capture_id", "exposure_ms": "neutral_default_units_unverified",
                                "analog_gain": "neutral_default", "noise_sigma": "estimated_per_patch",
                                "mean_luma": "estimated_per_patch"}
    }
    return meta


class PairedRawDataset(Dataset):
    """Random patches from verified SIDD-CC train pairs.

    root points to the SIDD-CC dataset root containing Raw/train and SRGB/train.
    patch_size is the full-resolution RGB output size and must be divisible by 2.
    """
    def __init__(self, root, split="train", patch_size=256, validation_fraction=0.1,
                 seed=42, patches_per_capture=8):
        self.root = Path(root)
        self.patch_size = int(patch_size)
        if self.patch_size < 32 or self.patch_size % 2:
            raise ValueError("patch_size must be an even integer >= 32")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        raw_root = self.root / "Raw" / "train" / "Data"
        rgb_root = self.root / "SRGB" / "train" / "Data"
        if not raw_root.is_dir() or not rgb_root.is_dir():
            raise FileNotFoundError(f"Expected {raw_root} and {rgb_root}")
        instances = sorted(p.name for p in raw_root.iterdir() if p.is_dir())
        if len(instances) < 2:
            raise RuntimeError("Need at least two instance directories for instance-level split")
        rng = random.Random(seed)
        shuffled = instances[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * validation_fraction))
        val_instances = set(shuffled[:n_val])
        selected = val_instances if split == "val" else set(instances) - val_instances
        self.records = []
        for instance in sorted(selected):
            rdir, sdir = raw_root / instance, rgb_root / instance
            for noisy in sorted(rdir.glob("*_NOISY_RAW_*.MAT")):
                # Preserve the capture prefix and index exactly.
                prefix, index = noisy.name.rsplit("_NOISY_RAW_", 1)
                index = index[:-4]
                gt = rdir / f"{prefix}_GT_RAW_{index}.MAT"
                metadata = rdir / f"{prefix}_METADATA_RAW_{index}.MAT"
                rgb_gt = sdir / f"{prefix}_GT_SRGB_{index}.PNG"
                if not all(p.is_file() for p in (gt, metadata, rgb_gt)):
                    raise FileNotFoundError(f"Incomplete pair for {noisy}")
                self.records.append((instance, noisy, gt, metadata, rgb_gt))
        if not self.records:
            raise RuntimeError(f"No paired captures found for split={split}")
        self.patches_per_capture = int(patches_per_capture)
        self._deterministic = split == "val"

    def __len__(self):
        return len(self.records) * self.patches_per_capture

    def __getitem__(self, index):
        instance, noisy_path, gt_path, meta_path, rgb_path = self.records[index % len(self.records)]
        # Captures are sampled contiguously by CaptureBatchSampler. Keep only
        # the current capture decoded: MAT and PNG decoding is expensive, while
        # retaining the entire 32.6 GB dataset in RAM is not viable.
        cache_key = str(noisy_path)
        if getattr(self, "_cache_key", None) != cache_key:
            noisy = np.asarray(_mat_value(noisy_path, "data"), dtype=np.float32).squeeze()
            gt = np.asarray(_mat_value(gt_path, "data"), dtype=np.float32).squeeze()
            if noisy.ndim != 2 or gt.shape != noisy.shape:
                raise ValueError(f"RAW shape mismatch for {noisy_path}")
            with Image.open(rgb_path) as im:
                target = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            if target.shape[:2] != noisy.shape:
                raise ValueError(f"RAW/RGB dimensions mismatch for {noisy_path}: {noisy.shape} vs {target.shape}")
            meta = _read_metadata(meta_path, instance)
            self._cache_key = cache_key
            self._cache_noisy = noisy
            self._cache_gt = gt
            self._cache_target = target
            self._cache_meta = meta
        noisy = self._cache_noisy
        gt = self._cache_gt
        target = self._cache_target
        meta = dict(self._cache_meta)
        # The requested patch size is measured in full-resolution sensor pixels.
        # pack_bayer halves both spatial dimensions, while the RGB target remains
        # full resolution. Bound the crop using the FULL patch size (not half).
        max_y = noisy.shape[0] - self.patch_size
        max_x = noisy.shape[1] - self.patch_size
        if max_y < 0 or max_x < 0:
            raise ValueError(
                f"Patch size {self.patch_size} exceeds source RAW dimensions {noisy.shape}"
            )
        # Bayer packing requires even-aligned origins. Sampling in units of two
        # guarantees a complete patch and exact RAW/RGB spatial correspondence.
        if getattr(self, "_deterministic", False):
            rng = random.Random(104729 + index)
        else:
            rng = random
        y = 2 * rng.randrange(max_y // 2 + 1)
        x = 2 * rng.randrange(max_x // 2 + 1)
        raw_patch = noisy[y:y+self.patch_size, x:x+self.patch_size]
        gt_patch = gt[y:y+self.patch_size, x:x+self.patch_size]
        rgb_patch = target[y:y+self.patch_size, x:x+self.patch_size]
        if raw_patch.shape != (self.patch_size, self.patch_size):
            raise RuntimeError("Unexpected RAW patch shape")
        # Use RAW patch statistics as explicit estimated conditioning features.
        meta["noise_sigma"] = float(np.std(raw_patch - gt_patch))
        meta["mean_luma"] = float(np.clip(rgb_patch.mean(), 0, 1))
        packed = pack_bayer(raw_patch, meta["pattern"]).astype(np.float32)
        return {
            "raw": torch.from_numpy(packed.transpose(2, 0, 1).copy()),
            "metadata": torch.from_numpy(metadata_vector(meta)),
            "target_srgb": torch.from_numpy(rgb_patch.transpose(2, 0, 1).copy()),
            "instance": instance,
            "capture": noisy_path.stem,
        }
