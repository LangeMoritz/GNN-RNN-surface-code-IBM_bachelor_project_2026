"""
Two-phase fine-tuning:
  Phase A — train on DEM-sampled data with DEM val set (early stop).
  Phase B — continue training on real hardware shots with real val set.

Final evaluation on the same held-out real test split used by the DEM-only
and real-only scripts (same seed=42, same 75/15/15 split).
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch

from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import split_ibm_job, evaluate_dataset
from dem_dataset import DEMDataset
from build_dem_from_detection_events import build_dem_from_detection_events
from stim_alignment import build_stim_alignment, ibm_detections_to_stim_order
from utils import TrainingLogger


D, T = 3, 10
JOB = "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json"

PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_ibm_dem_real"
PATIENCE_A = 30
PATIENCE_B = 50


# Phase A (DEM) and Phase B (real) share distance/dt/batch but differ in
# learning rate and epochs — real phase fine-tunes more gently.
args_dem = Args(
    distance=D,
    dt=2,
    batch_size=256,
    n_batches=128,
    n_epochs=200,
    lr=3e-4,
    min_lr=1e-5,
)
# Phase B — batch_size * n_batches >= 70k so every real training shot is seen
# at least once per epoch (70k train split, 1024 * 70 = 71_680).
args_real = Args(
    distance=D,
    dt=2,
    batch_size=1024,
    n_batches=70,
    n_epochs=200,
    lr=5e-5,
    min_lr=1e-6,
)

# --- Split real shots
sc = SurfaceCodeCircuit(distance=D, T=T)
real_train, real_val, real_test = split_ibm_job(
    sc, JOB, ratios=[0.70, 0.15, 0.15], seed=42,
    dt=args_real.dt, k=args_real.k, batch_size=args_real.batch_size, device=args_real.device,
)
print(f"Real shots — train: {len(real_train.logical_flips)}, "
      f"val: {len(real_val.logical_flips)}, test: {len(real_test.logical_flips)}")

# --- DEM calibrated from the 75% train split
alignment = build_stim_alignment(sc, rounds=T)
det_stim = ibm_detections_to_stim_order(
    real_train.detections, alignment.ibm_middle_order, alignment.ibm_z_order,
)
print(f"Calibrating DEM from {len(det_stim)} train shots...")
dem = build_dem_from_detection_events(alignment.circuit, det_stim)

dem_train = DEMDataset(args_dem, dem=dem, rounds=T, circuit=alignment.circuit,
                       detector_is_z=alignment.detector_is_z)
# dem_val = DEMDataset(args_dem, dem=dem, rounds=T, circuit=alignment.circuit,
#                      detector_is_z=alignment.detector_is_z)

# --- Model + phase A
model = GRUDecoder(args_dem)
ckpt = torch.load(PRETRAINED, weights_only=False, map_location=args_dem.device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.to(args_dem.device)

print("\n=== Phase A: DEM fine-tune ===")
logger_a = TrainingLogger(statsfile="finetune_dem_real_phaseA")
model.train_model(
    dataset=dem_train, val_dataset=real_val,
    n_val_batches=30, patience=PATIENCE_A,
    logger=logger_a,
)

# --- Phase B: continue on real hardware shots
# Swap model args so the new optimizer uses the real-phase LR schedule.
model.args = args_real

print("\n=== Phase B: real-data fine-tune ===")
logger_b = TrainingLogger(statsfile="finetune_dem_real_phaseB")
model.train_model(
    dataset=real_train, val_dataset=real_val,
    n_val_batches=30, patience=PATIENCE_B,
    save=SAVE_NAME, logger=logger_b,
)

# --- Final evaluation on held-out real test
real_test_m = evaluate_dataset(model, real_test, n_batches=40)
acc = real_test_m["acc"]
lfr_round = 1.0 - acc ** (1.0 / T) if acc > 0 else 1.0
print(f"\nReal test:")
print(f" acc = {acc:.4f}  (c0={real_test_m['acc_0']:.4f}, c1={real_test_m['acc_1']:.4f})")
print(f" shots = {real_test_m['n_0'] + real_test_m['n_1']}  "
      f"(class-0: {real_test_m['n_0']}, class-1: {real_test_m['n_1']})")
print(f" LFR/round = {lfr_round:.4f}  (1 - acc^(1/T) with T={T})")
