#!/usr/bin/env python3
"""
Run CBSS decomposition on the first isometric phase (0–5 s) of every noisy
FDSI recording.

The first 5 s of every contraction is at 0 ° wrist angle with constant 15 % MVC,
which constitutes a genuine isometric phase regardless of the overall movement
pattern.  CBSS is therefore applied to samples [0, 5 s) of each noisy recording.

Reads  : outputs/fdsi_benchmark/data/<sub>/noisy/<prefix>_snr<N>dB_emg.npz
Writes : outputs/fdsi_benchmark/decomposition/<sub>/<prefix>_snr<N>dB_cbss.npz
                                               <sub>/<prefix>_snr<N>dB_cbss_log.json

Note   : with 320 channels and extension factor 12, each CBSS call may take
         5–15 min.  Total wall time for 100 decompositions ≈ 8–25 h.
         Consider reducing ext_fact in configs/cbss.json for quick experiments.

Run (inside container or muniverse env):
    python scripts/run_decomposition_fdsi.py --output_dir outputs/fdsi_benchmark
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_repo_root, os.path.join(_repo_root, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from muniverse.algorithms.decomposition import decompose_cbss, load_config

SUBJECT_SEEDS  = [0, 1, 2, 3, 4]
MUSCLE         = "FDSI"
RAMP_DURATIONS = [40, 20, 10, 5]
CONDITIONS     = [f"triangular-ramp{r}s" for r in RAMP_DURATIONS] + ["staircase"]
SNR_LEVELS_DB  = [30, 25, 20, 15]
ISO_END_S      = 5.0    # length of the first isometric window to decompose
FS             = 2048


def subject_label(seed: int) -> str:
    return f"sub-{seed + 1:02d}"


def load_cbss_config_for_isometric(config_path: str, iso_end_s: float) -> dict:
    """Load CBSS config and restrict analysis window to the first isometric phase."""
    cfg = copy.deepcopy(load_config(config_path))
    cfg["Config"]["start_time"] = 0
    cfg["Config"]["end_time"]   = iso_end_s
    return cfg


def save_cbss_results(out_path: str, results: dict) -> None:
    """
    Save sources (n_units × n_samples), silhouette, and spike trains to a single
    compressed .npz.  Spike trains are stored as individual arrays named
    spikes_unit<id> so they can be loaded without allow_pickle.
    """
    save_dict: dict = {}
    sources = results.get("sources")
    sil     = results.get("silhouette")
    spikes  = results.get("spikes") or {}

    save_dict["sources"]   = sources if sources is not None else np.empty((0, 0), dtype=np.float32)
    save_dict["silhouette"] = sil   if sil     is not None else np.array([], dtype=np.float64)
    save_dict["n_units"]   = np.array([len(spikes)], dtype=np.int32)

    for uid, sp in spikes.items():
        save_dict[f"spikes_unit{uid}"] = np.asarray(sp, dtype=np.int64)

    np.savez_compressed(out_path, **save_dict)


def main(output_dir: str) -> None:
    cbss_cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "cbss.json",
    )
    cbss_cfg = load_cbss_config_for_isometric(cbss_cfg_path, ISO_END_S)

    total   = len(SUBJECT_SEEDS) * len(CONDITIONS) * len(SNR_LEVELS_DB)
    done    = 0
    t_start = time.time()

    for seed in SUBJECT_SEEDS:
        sub_id  = subject_label(seed)
        src_dir = os.path.join(output_dir, "data", sub_id, "noisy")
        dec_dir = os.path.join(output_dir, "decomposition", sub_id)
        os.makedirs(dec_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  {sub_id}")
        print(f"{'=' * 60}")

        for cond_label in CONDITIONS:
            for snr_db in SNR_LEVELS_DB:
                tag    = f"snr{snr_db}dB"
                prefix = f"{sub_id}_{MUSCLE}_{cond_label}_{tag}"
                emg_path = os.path.join(src_dir, f"{prefix}_emg.npz")
                out_npz  = os.path.join(dec_dir, f"{prefix}_cbss.npz")
                out_log  = os.path.join(dec_dir, f"{prefix}_cbss_log.json")

                if not os.path.exists(emg_path):
                    print(f"  MISSING {emg_path}")
                    continue

                if os.path.exists(out_npz):
                    print(f"  SKIP (exists)  {prefix}")
                    done += 1
                    continue

                print(f"\n  → {cond_label}  {tag}", end="", flush=True)
                t0 = time.time()

                emg_noisy = np.load(emg_path)["emg"].astype(np.float32)
                # CBSS expects (channels × samples)
                data = emg_noisy.T

                results, log_data = decompose_cbss(
                    data             = data,
                    algorithm_config = cbss_cfg,
                    metadata         = {
                        "filename": f"{prefix}_emg.npz",
                        "format":   "npz",
                    },
                )

                n_mus = len(results.get("spikes") or {})
                elapsed = time.time() - t0
                print(f"  →  {n_mus} MUs  ({elapsed:.0f} s)")

                save_cbss_results(out_npz, results)

                # Augment log with provenance
                log_data["decomposition_meta"] = {
                    "subject_id":    sub_id,
                    "muscle":        MUSCLE,
                    "condition":     cond_label,
                    "snr_db":        snr_db,
                    "iso_window_s":  [0, ISO_END_S],
                    "n_identified":  n_mus,
                    "elapsed_s":     round(elapsed, 1),
                }
                with open(out_log, "w") as fh:
                    json.dump(log_data, fh, indent=2, default=str)

                done += 1
                elapsed_total = time.time() - t_start
                rate = done / elapsed_total if elapsed_total > 0 else 0
                eta  = (total - done) / rate if rate > 0 else float("nan")
                print(
                    f"     [{done}/{total}]  "
                    f"elapsed {elapsed_total/60:.1f} min  "
                    f"ETA {eta/60:.1f} min"
                )

    print(f"\n✓  Decomposition complete.  {done}/{total} recordings processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CBSS on the first isometric phase of FDSI noisy recordings"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/fdsi_benchmark",
        help="Root output directory (default: outputs/fdsi_benchmark)",
    )
    args = parser.parse_args()
    main(args.output_dir)
