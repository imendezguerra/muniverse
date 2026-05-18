#!/usr/bin/env python3
"""
Inject Gaussian noise into clean FDSI recordings at four SNR levels.

Reads  : outputs/fdsi_benchmark/data/<sub>/clean/<prefix>_emg.npz
Writes : outputs/fdsi_benchmark/data/<sub>/noisy/<prefix>_snr<N>dB_emg.npz

SNR formula (matched to _run_neuromotion.py):
    std_noise = std(EMG_clean) * 10^(-SNR_dB / 20)

Run (no container needed):
    python scripts/inject_noise_fdsi.py --output_dir outputs/fdsi_benchmark
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SNR_LEVELS_DB = [30, 25, 20, 15]
SUBJECT_SEEDS = [0, 1, 2, 3, 4]
MUSCLE        = "FDSI"
RAMP_DURATIONS = [40, 20, 10, 5]
CONDITIONS     = [f"triangular-ramp{r}s" for r in RAMP_DURATIONS] + ["staircase"]


def subject_label(seed: int) -> str:
    return f"sub-{seed + 1:02d}"


def add_noise(emg_clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise scaled to achieve the target SNR."""
    std_emg   = emg_clean.std()
    std_noise = std_emg * 10 ** (-snr_db / 20.0)
    return emg_clean + rng.normal(0.0, std_noise, emg_clean.shape).astype(emg_clean.dtype)


def achieved_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise = noisy - clean
    if noise.std() == 0:
        return float("inf")
    return 20.0 * np.log10(clean.std() / noise.std())


def main(output_dir: str) -> None:
    for seed in SUBJECT_SEEDS:
        sub_id   = subject_label(seed)
        src_dir  = os.path.join(output_dir, "data", sub_id, "clean")
        dst_dir  = os.path.join(output_dir, "data", sub_id, "noisy")
        os.makedirs(dst_dir, exist_ok=True)

        print(f"\n{sub_id}")
        for cond_idx, cond_label in enumerate(CONDITIONS):
            emg_path = os.path.join(src_dir, f"{sub_id}_{MUSCLE}_{cond_label}_emg.npz")
            if not os.path.exists(emg_path):
                print(f"  MISSING  {emg_path}")
                continue

            emg_clean = np.load(emg_path)["emg"].astype(np.float32)

            for snr_idx, snr_db in enumerate(SNR_LEVELS_DB):
                # Deterministic noise seed: subject × 10000 + condition × 100 + snr index
                noise_seed = seed * 10000 + cond_idx * 100 + snr_idx
                rng        = np.random.default_rng(noise_seed)
                emg_noisy  = add_noise(emg_clean, snr_db, rng)

                tag     = f"snr{snr_db}dB"
                prefix  = f"{sub_id}_{MUSCLE}_{cond_label}_{tag}"
                out_emg = os.path.join(dst_dir, f"{prefix}_emg.npz")
                np.savez_compressed(out_emg, emg=emg_noisy)

                # Companion metadata
                meta = {
                    "subject_id":    sub_id,
                    "muscle":        MUSCLE,
                    "condition":     cond_label,
                    "snr_db_target": snr_db,
                    "snr_db_actual": round(achieved_snr(emg_clean, emg_noisy), 2),
                    "noise_seed":    noise_seed,
                    "emg_shape":     list(emg_noisy.shape),
                }
                with open(
                    os.path.join(dst_dir, f"{prefix}_noise_metadata.json"), "w"
                ) as fh:
                    json.dump(meta, fh, indent=2)

                print(
                    f"  {cond_label:28s}  {tag}  "
                    f"actual SNR={meta['snr_db_actual']:.1f} dB  {emg_noisy.shape}"
                )

    print("\n✓  Noise injection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject noise into FDSI clean recordings")
    parser.add_argument(
        "--output_dir",
        default="outputs/fdsi_benchmark",
        help="Root output directory (default: outputs/fdsi_benchmark)",
    )
    args = parser.parse_args()
    main(args.output_dir)
