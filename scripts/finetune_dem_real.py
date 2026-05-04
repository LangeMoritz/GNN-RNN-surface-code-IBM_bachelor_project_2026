"""
Two-phase fine-tuning:
  Phase A — train on DEM-sampled data with real val set (early stop).
  Phase B — continue training on real hardware shots with real val set.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import prepare_real_datasets, evaluate_dataset
from mwpm_decoder import evaluate_mwpm_split
from dem_dataset import DEMDataset
from build_dem_from_detection_events import build_dem_from_detection_events
from stim_alignment import build_stim_alignment, ibm_detections_to_stim_order
from utils import TrainingLogger, print_test_result

D, T = 5, 20
TRAIN_JOBS = [
    "jobs/dist5/d5_T20_shots150000_d7s7584t738s73cgb1u0.json",
    "jobs/dist5/job_d5_T20_shots50000_d7fmn4l6agrc738ispv0.json",
]

PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_dem_real_t20_c"
STATSFILE_NAME=F"d{D}_dem_real_t20_c"
PATIENCE_A = 20
PATIENCE_B = 60

# Phase A (DEM-sampled)
args_dem = Args(
    distance=D,
    dt=2,
    batch_size=256,
    n_batches=800,
    n_epochs=120,
    lr=2.5e-5,
    min_lr=5e-7,
)
# Phase B (real samples)
args_real = Args(
    distance=D,
    dt=2,
    batch_size=64,
    n_batches=2500,
    n_epochs=200,
    lr=1e-5,
    min_lr=5e-7,
)

# Build train/val/test from TRAIN_JOBS
sc = SurfaceCodeCircuit(distance=D, T=T)
real_train, real_val, real_test = prepare_real_datasets(
    sc, TRAIN_JOBS,
    dt=args_real.dt, k=args_real.k,
    batch_size=args_real.batch_size, device=args_real.device,
)

# DEM calibrated from ALL train+val detections (test excluded)
alignment = build_stim_alignment(sc, rounds=T)
all_train_det = np.concatenate(
    [real_train.detections, real_val.detections], axis=0,
)
det_stim = ibm_detections_to_stim_order(
    all_train_det, alignment.ibm_middle_order, alignment.ibm_z_order,
)
print(f"Calibrating DEM from {len(det_stim)} shots (all train+val, test excluded)")
dem = build_dem_from_detection_events(alignment.circuit, det_stim)

dem_train = DEMDataset(
    args_dem, dem=dem, rounds=T, circuit=alignment.circuit,
    detector_is_z=alignment.detector_is_z
)

# Model + Phase A
model = GRUDecoder(args_dem)
ckpt = torch.load(PRETRAINED, weights_only=False, map_location=args_dem.device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.to(args_dem.device)

print("\n=== Phase A: DEM fine-tune ===")
logger_a = TrainingLogger(
    logfile=f"{SAVE_NAME}.log",
    statsfile=f"{STATSFILE_NAME}_phaseA",
)
model.train_model(
    dataset=dem_train, val_dataset=real_val,
    n_val_batches=300, patience=PATIENCE_A,
    logger=logger_a
)

# Phase B: continue on real hardware shots
# Swap model args so the new optimizer uses the real-phase LR schedule.
model.args = args_real

print("\n=== Phase B: real-data fine-tune ===")
logger_b = TrainingLogger(
    logfile=f"{SAVE_NAME}.log",
    statsfile=f"{STATSFILE_NAME}_phaseB",
)
model.train_model(
    dataset=real_train, val_dataset=real_val,
    n_val_batches=300, patience=PATIENCE_B,
    save=SAVE_NAME, logger=logger_b
)

# Final evaluation on held-out real test
real_test_m = evaluate_dataset(model, real_test, all_shots=True)
print_test_result(real_test_m, T)

mwpm_test_m = evaluate_mwpm_split(sc, [real_train, real_val], real_test)
print_test_result(mwpm_test_m, T, label="MWPM real test")
