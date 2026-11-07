import numpy as np
import torch
from scipy.io import savemat
from PIL import Image

from datasets.loader import PairedRawDataset


def test_dataset_patch_is_full_size_and_even_aligned(tmp_path):
    root = tmp_path / "SIDD-CC"
    for instance_no in (1, 2):
        instance = f"{instance_no:04d}_001_S6_003200_00060_3200_L"
        raw_dir = root / "Raw" / "train" / "Data" / instance
        rgb_dir = root / "SRGB" / "train" / "Data" / instance
        raw_dir.mkdir(parents=True)
        rgb_dir.mkdir(parents=True)
        prefix, idx = instance, "034"
        noisy = np.random.default_rng(instance_no).random((300, 400), dtype=np.float32)
        clean = np.random.default_rng(instance_no + 10).random((300, 400), dtype=np.float32)
        savemat(raw_dir / f"{prefix}_NOISY_RAW_{idx}.MAT", {"data": noisy})
        savemat(raw_dir / f"{prefix}_GT_RAW_{idx}.MAT", {"data": clean})
        savemat(raw_dir / f"{prefix}_METADATA_RAW_{idx}.MAT", {
            "cfa_str": np.array(list("rggb"), dtype="U1")
        })
        rgb = np.random.default_rng(instance_no + 20).integers(0, 256, (300, 400, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(rgb_dir / f"{prefix}_GT_SRGB_{idx}.PNG")

    ds = PairedRawDataset(root, split="train", patch_size=128,
                          validation_fraction=0.5, patches_per_capture=4)
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["raw"].shape == (4, 64, 64)
        assert sample["target_srgb"].shape == (3, 128, 128)
        assert sample["metadata"].shape == (5,)
        assert torch.isfinite(sample["raw"]).all()
