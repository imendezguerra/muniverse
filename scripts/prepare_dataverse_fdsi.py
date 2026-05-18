#!/usr/bin/env python3
"""
Prepare the FDSI benchmark dataset for upload to Harvard Dataverse.

Creates outputs/fdsi_benchmark/dataverse_upload/ containing:
  - README.md           dataset description, file conventions, loading instructions
  - manifest.csv        file list with sizes (bytes) and MD5 checksums
  - data/               copy of all clean + noisy EMG recordings
  - decomposition/      copy of all CBSS results

Run:
    python scripts/prepare_dataverse_fdsi.py --output_dir outputs/fdsi_benchmark
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

SUBJECT_SEEDS  = [0, 1, 2, 3, 4]
MUSCLE         = "FDSI"
RAMP_DURATIONS = [40, 20, 10, 5]
CONDITIONS     = [f"triangular-ramp{r}s" for r in RAMP_DURATIONS] + ["staircase"]
SNR_LEVELS_DB  = [30, 25, 20, 15]


def subject_label(seed: int) -> str:
    return f"sub-{seed + 1:02d}"


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


README_TEMPLATE = """\
# FDSI Dynamic Contraction Benchmark Dataset

**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
**Generated with**: MUniverse (GPL-3.0) · BioMime (GPL-3.0) · NeuroMotion (GPL-3.0)
**Date generated**: {date}

---

## Overview

Synthetic high-density surface EMG recordings of the Flexor Digitorum Superficialis
(FDSI) during dynamic wrist contractions, with known ground-truth motor-unit spike
trains.  Intended for benchmarking EMG decomposition algorithms.

## Experimental design

| Parameter | Value |
|---|---|
| Muscle | FDSI (Flexor Digitorum Superficialis I) |
| Movement | Wrist Flexion-Extension |
| Effort | Constant 15 % MVC |
| Electrode grid | 10 rows × 32 columns (320 channels), 8 mm i.e.d. |
| Sampling rate | 2048 Hz |
| Subjects (simulated) | 5 (seeds 0–4) |

### Contraction conditions

| Label | Type | Description |
|---|---|---|
| `triangular-ramp40s` | Triangular dynamic | 5 s rest → 1 × 40 s ramp-up/down → 5 s rest (90 s) |
| `triangular-ramp20s` | Triangular dynamic | 5 s rest → 2 × 20 s ramp-up/down → 5 s rest (90 s) |
| `triangular-ramp10s` | Triangular dynamic | 5 s rest → 4 × 10 s ramp-up/down → 5 s rest (90 s) |
| `triangular-ramp5s`  | Triangular dynamic | 5 s rest → 8 × 5 s ramp-up/down → 5 s rest (90 s) |
| `staircase`          | Staircase dynamic | 0 → 40 ° in 10 ° steps (5 s holds, 10 s ramps) (125 s) |

All conditions start and end with 5 s at 0 ° (first isometric phase suitable for
decomposition benchmarking).

### SNR levels

Clean EMG is provided alongside four noisy versions at 30, 25, 20, and 15 dB SNR
(Gaussian noise scaled to the RMS of the clean signal).

---

## File naming convention

```
data/<sub-id>/clean/<sub-id>_FDSI_<condition>_emg.npz       ← clean EMG
data/<sub-id>/clean/<sub-id>_FDSI_<condition>_spikes.npz    ← ground-truth spikes
data/<sub-id>/clean/<sub-id>_FDSI_<condition>_angle.npz     ← wrist angle profile (deg)
data/<sub-id>/clean/<sub-id>_FDSI_<condition>_effort.npz    ← effort profile (0–1)
data/<sub-id>/clean/<sub-id>_FDSI_<condition>_metadata.json ← recording parameters

data/<sub-id>/noisy/<sub-id>_FDSI_<condition>_snr<N>dB_emg.npz
data/<sub-id>/noisy/<sub-id>_FDSI_<condition>_snr<N>dB_noise_metadata.json

decomposition/<sub-id>/<sub-id>_FDSI_<condition>_snr<N>dB_cbss.npz
decomposition/<sub-id>/<sub-id>_FDSI_<condition>_snr<N>dB_cbss_log.json
```

---

## Array shapes and keys

| File | Key | Shape |
|---|---|---|
| `*_emg.npz` | `emg` | (n_samples, 320) |
| `*_spikes.npz` | `spikes` | object array, one entry per MU (sample indices) |
| `*_angle.npz` | `angle` | (n_samples,) degrees |
| `*_effort.npz` | `effort` | (n_samples,) normalised 0–1 |
| `*_cbss.npz` | `sources` | (n_identified_MUs, n_iso_samples) |
| `*_cbss.npz` | `silhouette` | (n_identified_MUs,) |
| `*_cbss.npz` | `spikes_unit<id>` | (n_spikes,) sample indices within iso window |

Channel ordering in `emg`: channel `k` → row `k // 32`, column `k % 32`.

---

## Loading example (Python)

```python
import numpy as np

# Clean EMG
emg = np.load("data/sub-01/clean/sub-01_FDSI_triangular-ramp40s_emg.npz")["emg"]
# emg.shape → (184320, 320)  [90 s × 2048 Hz]

# Ground-truth spikes for MU 0 (sample indices)
spikes_obj = np.load(
    "data/sub-01/clean/sub-01_FDSI_triangular-ramp40s_spikes.npz",
    allow_pickle=True,
)["spikes"]
gt_spikes_mu0 = spikes_obj[0]          # sample indices
gt_spikes_mu0_sec = gt_spikes_mu0 / 2048  # convert to seconds

# CBSS decomposition at 20 dB SNR
cbss = np.load(
    "decomposition/sub-01/sub-01_FDSI_triangular-ramp40s_snr20dB_cbss.npz"
)
sources   = cbss["sources"]    # (n_MUs, 10240)  — first 5 s window
silhouette = cbss["silhouette"]
est_spikes = {
    int(k.replace("spikes_unit", "")): cbss[k]
    for k in cbss.files if k.startswith("spikes_unit")
}
```

---

## Loading via easyDataverse

```python
from easyDataverse import Dataverse
dataverse = Dataverse(server_url="https://dataverse.harvard.edu/")
dataset   = dataverse.load_dataset(
    pid="doi:10.7910/DVN/XXXXXXX",   # replace with actual DOI after upload
    filedir="./fdsi_data/",
)
```

---

## Citation

If you use this dataset please cite:

> [MUniverse paper / preprint — add reference here]

and the underlying simulation tools:

> Ma, S. et al. (2022). "BioMime: A Generative Model for Biosystem Simulation".
> arXiv:2211.01856

> [NeuroMotion reference — add here]

---

## Contact

[Add your name and email here]
"""


def collect_files(src_root: str) -> list[dict]:
    """Walk src_root and return sorted list of file records."""
    records = []
    for dirpath, _, filenames in os.walk(src_root):
        for fname in sorted(filenames):
            fpath = os.path.join(dirpath, fname)
            rel   = os.path.relpath(fpath, src_root)
            records.append({
                "relative_path": rel,
                "size_bytes":    os.path.getsize(fpath),
                "md5":           md5sum(fpath),
            })
    return records


def copy_tree_filtered(src: str, dst: str) -> None:
    """Copy src → dst, skipping the dataverse_upload dir to avoid recursion."""
    for item in Path(src).iterdir():
        if item.name == "dataverse_upload":
            continue
        dest = Path(dst) / item.name
        if item.is_dir():
            shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
        else:
            shutil.copy2(str(item), str(dest))


def main(output_dir: str) -> None:
    upload_dir = os.path.join(output_dir, "dataverse_upload")
    os.makedirs(upload_dir, exist_ok=True)

    # ── Copy data and decomposition trees ─────────────────────────────────
    for subdir in ("data", "decomposition"):
        src = os.path.join(output_dir, subdir)
        dst = os.path.join(upload_dir, subdir)
        if os.path.exists(src):
            print(f"Copying {subdir}/ …")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            print(f"  WARNING: {src} not found — run generation/noise/decomposition first.")

    # ── README.md ─────────────────────────────────────────────────────────
    readme_path = os.path.join(upload_dir, "README.md")
    with open(readme_path, "w") as fh:
        fh.write(README_TEMPLATE.format(date=time.strftime("%Y-%m-%d")))
    print(f"Written: README.md")

    # ── manifest.csv ──────────────────────────────────────────────────────
    manifest_path = os.path.join(upload_dir, "manifest.csv")
    print("Building manifest (computing MD5 checksums) …")
    records = collect_files(upload_dir)
    # Exclude manifest itself if it was left from a previous run
    records = [r for r in records if not r["relative_path"].endswith("manifest.csv")]

    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "size_bytes", "md5"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Written: manifest.csv  ({len(records)} files)")

    # ── Summary ───────────────────────────────────────────────────────────
    total_bytes = sum(r["size_bytes"] for r in records)
    print(f"\n✓  Upload package ready at: {upload_dir}")
    print(f"   Files : {len(records)}")
    print(f"   Total size : {total_bytes / 1e9:.2f} GB")
    print(
        "\nNext steps:"
        "\n  1. Review and update README.md (add your name, DOI placeholders)"
        "\n  2. Create a new dataset at https://dataverse.harvard.edu/dataverse/muniverse-datasets"
        "\n  3. Upload files via the web UI or easyDataverse API"
        "\n  4. Set license to CC BY 4.0"
        "\n  5. Publish and add the assigned DOI to your README and code"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare FDSI benchmark dataset for Harvard Dataverse upload"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/fdsi_benchmark",
        help="Root output directory (default: outputs/fdsi_benchmark)",
    )
    args = parser.parse_args()
    main(args.output_dir)
