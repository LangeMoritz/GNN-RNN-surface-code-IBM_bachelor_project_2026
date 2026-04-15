"""
Two-phase fine-tuning:
  Phase A — train on DEM-sampled data with DEM val set (early stop).
  Phase B — continue training on real hardware shots with real val set.

Final evaluation on the same held-out real test split used by the DEM-only
and real-only scripts (same seed=42, same 75/15/15 split).
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import copy
import torch

from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import split_ibm_job, evaluate_dataset
from dem_dataset import DEMDataset
from build_dem_from_detection_events import build_dem_from_detection_events
from stim_alignment import build_stim_alignment, ibm_detections_to_stim_order
from utils import TrainingLogger


D, T = 3, 20
JOB = "jobs/dist3/job_d3_T20_shots50000_d7fmgem2cugc739qov6g.json"
PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_ibm_dem_real"

# Phase A (DEM) and Phase B (real) share distance/dt/batch but differ in
# learning rate and epochs — real phase fine-tunes more gently.
args_dem = Args(
    distance=D, dt=2, batch_size=512, n_batches=64,
    n_epochs=800, lr=1e-3, min_lr=1e-5,
)
args_real = Args(
    distance=D, dt=2, batch_size=512, n_batches=32,
    n_epochs=400, lr=1e-5, min_lr=1e-7,
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
dem_val = DEMDataset(args_dem, dem=dem, rounds=T, circuit=alignment.circuit,
                     detector_is_z=alignment.detector_is_z)

# --- Model + phase A
model = GRUDecoder(args_dem)
ckpt = torch.load(PRETRAINED, weights_only=False, map_location=args_dem.device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.to(args_dem.device)

print("\n=== Phase A: DEM fine-tune ===")
logger_a = TrainingLogger(logfile="finetune_dem_real_phaseA.log",
                          statsfile="finetune_dem_real_phaseA")
model.train_model(
    dataset=dem_train, val_dataset=dem_val,
    n_val_batches=10, patience=50,
    save=f"{SAVE_NAME}_phaseA", logger=logger_a,
)

# --- Phase B: continue on real hardware shots
# Swap model args so the new optimizer uses the real-phase LR schedule.
model.args = args_real

print("\n=== Phase B: real-data fine-tune ===")
logger_b = TrainingLogger(logfile="finetune_dem_real_phaseB.log",
                          statsfile="finetune_dem_real_phaseB")
model.train_model(
    dataset=real_train, val_dataset=real_val,
    n_val_batches=10, patience=40,
    save=SAVE_NAME, logger=logger_b,
)

# --- Final evaluation on held-out real test
real_test_acc = evaluate_dataset(model, real_test, n_batches=40)
print(f"\nReal test accuracy: {real_test_acc:.4f}")
