#!/usr/bin/env python3
"""
Generate FDSI benchmark dataset — clean EMG only (noise added separately).

Conditions (all at constant 15 % MVC, Flexion-Extension DOF):
  Triangular dynamic  : ramp durations 40 s, 20 s, 10 s, 5 s
                        → 80 s of active movement + 5 s bookend rest at each end (90 s total)
  Staircase dynamic   : 0→40 deg in 10-deg steps, 5 s holds, 10 s ramps (125 s total)

Subjects : 5 (seeds 0–4)
Electrode grid : full 10 × 32 (320 channels), no column selection
MUAPs          : generated once per subject, cached to disk; reused for all conditions.

Run inside the muniverse neuromotion container:
    docker run --rm \\
      -v $(pwd):/workspace -w /workspace \\
      muniverse-test:neuromotion \\
      python scripts/generate_fdsi_dataset.py --output_dir outputs/fdsi_benchmark

Output layout:
    outputs/fdsi_benchmark/
    ├── cache/          ← MUAP .npy + metadata per subject
    └── data/
        └── sub-<N>/
            └── clean/  ← emg, spikes, angle, effort, metadata per condition
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from easydict import EasyDict as edict

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_repo_root, os.path.join(_repo_root, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from muniverse.data_generation._run_neuromotion import (
    build_movement_profile,
    cache_muaps,
    generate_emg_signal,
    generate_muaps,
    generate_spike_trains,
    load_cached_muaps,
    select_optimal_electrode_columns,
)
from muniverse.data_generation.generate_configs import (
    MUSCLE_LABELS,
    generate_subject_properties,
)
from NeuroMotion.MNPoollib.mn_params import mn_default_settings
from NeuroMotion.MNPoollib.MNPool import MotoneuronPool

# ── Fixed parameters ───────────────────────────────────────────────────────────
MUSCLE          = "FDSI"
MOVEMENT_DOF    = "Flexion-Extension"
FS              = 2048
EFFORT_LEVEL    = 0.15          # 15 % MVC (constant throughout every contraction)
SUBJECT_SEEDS   = [0, 1, 2, 3, 4]
PEAK_ANGLE_DEG  = 40.0          # maximum wrist angle (degrees)
BOOKEND_REST_S  = 5.0           # seconds at 0 ° before and after movement
ACTIVE_DUR_S    = 80.0          # seconds of angle movement (same for all triangular)
RAMP_DURATIONS  = [40, 20, 10, 5]

STAIRCASE_STEP_DEG = 10
STAIRCASE_MAX_DEG  = 40
STAIRCASE_HOLD_S   = 5.0        # also serves as the end bookend (hold at 0 °)
STAIRCASE_RAMP_S   = 10.0
DESIRED_COLS       = 10     # columns to select from 10×32 grid → 10×10 (100 ch)

MODEL_PTH  = "./ckp/model_linear.pth"
MUAP_PKL   = "./ckp/muap_examples.pkl"
MORPH      = False
FILTER_CFG = edict({"CutoffFrequency": 800, "FilterOrder": 4})
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
FDSI_IDX   = MUSCLE_LABELS.index(MUSCLE)   # position of FDSI in MUSCLE_LABELS list


# ── Condition registry ─────────────────────────────────────────────────────────
def _n_reps(ramp_s: int) -> int:
    return int(ACTIVE_DUR_S // (2 * ramp_s))

def _staircase_total_s(step_deg, max_deg, hold_s, ramp_s, bookend_s) -> float:
    n_transitions = max_deg // step_deg          # 4
    ascending  = n_transitions * (ramp_s + hold_s)
    descending = n_transitions * (ramp_s + hold_s)
    return bookend_s + ascending + descending     # 5+60+60 = 125 s

TRIANGULAR_CONDITIONS = {
    f"triangular-ramp{r}s": {
        "type":           "triangular",
        "ramp_duration_s": r,
        "n_reps":          _n_reps(r),
        "total_duration_s": BOOKEND_REST_S + ACTIVE_DUR_S + BOOKEND_REST_S,
    }
    for r in RAMP_DURATIONS
}
STAIRCASE_CONDITION = {
    "staircase": {
        "type":             "staircase",
        "step_deg":          STAIRCASE_STEP_DEG,
        "max_deg":           STAIRCASE_MAX_DEG,
        "hold_duration_s":   STAIRCASE_HOLD_S,
        "ramp_duration_s":   STAIRCASE_RAMP_S,
        "total_duration_s":  _staircase_total_s(
            STAIRCASE_STEP_DEG, STAIRCASE_MAX_DEG,
            STAIRCASE_HOLD_S, STAIRCASE_RAMP_S, BOOKEND_REST_S,
        ),
    }
}
ALL_CONDITIONS = {**TRIANGULAR_CONDITIONS, **STAIRCASE_CONDITION}


# ── Profile builders ───────────────────────────────────────────────────────────
def build_triangular_angle_profile(
    fs: int, ramp_s: float, n_reps: int, bookend_s: float, peak_deg: float
) -> np.ndarray:
    """Bookend rest → n_reps triangles (0 → peak → 0) → bookend rest."""
    ramp_n   = round(fs * ramp_s)
    rest_n   = round(fs * bookend_s)
    triangle = np.concatenate([
        np.linspace(0.0, peak_deg, ramp_n),
        np.linspace(peak_deg, 0.0, ramp_n),
    ])
    active = np.tile(triangle, n_reps)
    return np.concatenate([np.zeros(rest_n), active, np.zeros(rest_n)])


def build_staircase_angle_profile(
    fs: int,
    step_deg: float,
    max_deg: float,
    hold_s: float,
    ramp_s: float,
    bookend_s: float,
) -> np.ndarray:
    """
    5 s at 0 ° → ascending stairs (0 → max_deg in step_deg increments)
    → descending stairs (max_deg → 0) → 5 s at 0 ° (the final hold IS the bookend).
    """
    hold_n  = round(fs * hold_s)
    ramp_n  = round(fs * ramp_s)
    rest_n  = round(fs * bookend_s)
    steps   = np.arange(0.0, max_deg + step_deg, step_deg)   # [0, 10, 20, 30, 40]

    segs = [np.zeros(rest_n)]   # initial bookend at 0 °
    # Ascending: 0 → 10 → 20 → 30 → 40
    for i in range(len(steps) - 1):
        segs.append(np.linspace(steps[i], steps[i + 1], ramp_n))
        segs.append(np.full(hold_n, steps[i + 1]))
    # Descending: 40 → 30 → 20 → 10 → 0  (last hold at 0 ° = end bookend)
    for i in range(len(steps) - 1, 0, -1):
        segs.append(np.linspace(steps[i], steps[i - 1], ramp_n))
        segs.append(np.full(hold_n, steps[i - 1]))
    return np.concatenate(segs)


# ── I/O helpers ────────────────────────────────────────────────────────────────
def subject_label(seed: int) -> str:
    return f"sub-{seed + 1:02d}"


def clean_dir(output_dir: str, seed: int) -> str:
    return os.path.join(output_dir, "data", subject_label(seed), "clean")


def save_clean_recording(
    out_dir, subject_id, cond_label, emg, spikes, angle, effort, metadata
):
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{subject_id}_{MUSCLE}_{cond_label}"
    np.savez_compressed(os.path.join(out_dir, f"{prefix}_emg.npz"),    emg=emg)
    np.savez_compressed(
        os.path.join(out_dir, f"{prefix}_spikes.npz"),
        spikes=np.array(spikes, dtype=object),
    )
    np.savez_compressed(os.path.join(out_dir, f"{prefix}_angle.npz"),  angle=angle)
    np.savez_compressed(os.path.join(out_dir, f"{prefix}_effort.npz"), effort=effort)
    with open(os.path.join(out_dir, f"{prefix}_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"    saved  {prefix}   EMG {emg.shape}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(output_dir: str) -> None:
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\nOutput directory : {output_dir}")
    print(f"Subjects         : {[subject_label(s) for s in SUBJECT_SEEDS]}")
    print(f"Conditions       : {list(ALL_CONDITIONS.keys())}\n")

    for seed in SUBJECT_SEEDS:
        sub_id = subject_label(seed)
        print(f"{'=' * 60}")
        print(f"  {sub_id}  (seed={seed})")
        print(f"{'=' * 60}")

        # Subject-specific physiology
        fibre_density, mu_counts = generate_subject_properties(seed)
        num_mus = mu_counts[FDSI_IDX]
        print(f"  FDSI MUs: {num_mus}   fibre density: {fibre_density}")

        # ── MUAPs (cached per subject / muscle / DOF) ──────────────────────
        muap_cache = os.path.join(
            cache_dir, f"{sub_id}_{MUSCLE}_{MOVEMENT_DOF}_muaps.npy"
        )
        muap_meta = os.path.join(
            cache_dir, f"{sub_id}_{MUSCLE}_{MOVEMENT_DOF}_metadata.json"
        )
        required = {"muscle": MUSCLE, "fs": FS, "subject_id": sub_id}
        muaps, from_cache, cached_n, _ = load_cached_muaps(muap_cache, muap_meta, required)

        if not from_cache:
            print("  Generating MUAPs (full ROM, this takes a few minutes) …")
            movement_cfg = edict({
                "MovementDOF": MOVEMENT_DOF,
                "MovementType": "Dynamic",
                "MovementProfileParameters": edict(
                    {"AngleProfile": "Constant", "TargetAngle": 0}
                ),
            })
            poses, durations, _, steps = build_movement_profile(movement_cfg)
            muaps, num_mus, _ = generate_muaps(
                model_pth     = MODEL_PTH,
                ms_label      = MUSCLE,
                movement_cfg  = movement_cfg,
                fs_mov        = 50,
                poses         = poses,
                durations     = durations,
                steps         = steps,
                device        = DEVICE,
                morph         = MORPH,
                muap_file     = MUAP_PKL,
                fibre_density = fibre_density,
                fs            = FS,
                filter_cfg    = FILTER_CFG,
                num_mus       = num_mus,
                subject_seed  = seed,
            )
            cache_muaps(
                muaps, muap_cache,
                {**required, "num_mus": num_mus},
                muap_meta,
            )
            print(f"  MUAPs generated and cached: {muaps.shape}")
        else:
            num_mus = cached_n
            print(f"  MUAPs loaded from cache: {muaps.shape}")

        # Angle labels for the MUAP library (Flex-Ext: -65 → +65 °)
        muap_angle_labels = np.linspace(-65, 65, muaps.shape[1]).astype(int)
        out_clean = clean_dir(output_dir, seed)

        for cond_idx, (cond_label, cond_cfg) in enumerate(ALL_CONDITIONS.items()):
            print(f"\n  → {cond_label}")

            # Build angle profile
            if cond_cfg["type"] == "triangular":
                angle_profile = build_triangular_angle_profile(
                    FS,
                    cond_cfg["ramp_duration_s"],
                    cond_cfg["n_reps"],
                    BOOKEND_REST_S,
                    PEAK_ANGLE_DEG,
                )
            else:
                angle_profile = build_staircase_angle_profile(
                    FS,
                    STAIRCASE_STEP_DEG,
                    STAIRCASE_MAX_DEG,
                    STAIRCASE_HOLD_S,
                    STAIRCASE_RAMP_S,
                    BOOKEND_REST_S,
                )

            # Constant 15 % MVC throughout (including bookend periods)
            effort_profile = np.full(len(angle_profile), EFFORT_LEVEL, dtype=np.float32)
            actual_dur_s   = len(angle_profile) / FS

            # Reproducible spike trains (seed: subject × 100 + condition index)
            rng_seed = seed * 100 + cond_idx
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)
            mn_pool = MotoneuronPool(num_mus, MUSCLE, **mn_default_settings)
            _, spikes, _, _ = generate_spike_trains(mn_pool, effort_profile, FS)

            # Clean EMG — no noise added here
            emg_full = generate_emg_signal(
                muaps             = muaps,
                spikes            = spikes,
                time_samples      = len(effort_profile),
                muap_angle_labels = muap_angle_labels,
                angle_profile     = angle_profile,
                noise_level_db    = None,
                noise_seed        = None,
            )
            # emg_full shape: (n_samples, 320) — select 10×DESIRED_COLS centred on peak activity
            emg_clean, sel_cols, center_col = select_optimal_electrode_columns(
                emg_full, DESIRED_COLS
            )
            # emg_clean shape: (n_samples, 10*DESIRED_COLS)

            metadata = {
                "subject_id":           sub_id,
                "subject_seed":         seed,
                "muscle":               MUSCLE,
                "movement_dof":         MOVEMENT_DOF,
                "condition":            cond_label,
                "condition_type":       cond_cfg["type"],
                "effort_level_pct":     int(EFFORT_LEVEL * 100),
                "peak_angle_deg":       PEAK_ANGLE_DEG,
                "bookend_rest_s":       BOOKEND_REST_S,
                "total_duration_s":     round(actual_dur_s, 4),
                "isometric_phase_s":    {"start": 0.0, "end": BOOKEND_REST_S},
                "fs":                   FS,
                "n_channels":           emg_clean.shape[1],
                "n_rows":               10,
                "n_cols":               DESIRED_COLS,
                "selected_columns":     sel_cols,
                "center_column":        center_col,
                "full_grid_n_cols":     32,
                "n_motor_units":        num_mus,
                "fibre_density":        int(fibre_density),
                "rng_seed_spikes":      rng_seed,
                **{k: v for k, v in cond_cfg.items() if k != "type"},
                "generated_at":         time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_pth":            MODEL_PTH,
            }
            save_clean_recording(
                out_clean, sub_id, cond_label,
                emg_clean, spikes, angle_profile, effort_profile,
                metadata,
            )

    print("\n✓  Generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FDSI benchmark dataset")
    parser.add_argument(
        "--output_dir",
        default="outputs/fdsi_benchmark",
        help="Root output directory (default: outputs/fdsi_benchmark)",
    )
    args = parser.parse_args()
    main(args.output_dir)
