"""
Fine-tune on DEM-sampled data only, with train/val/test splits drawn
independently from the same calibrated DEM. Final evaluation on a held-out
real-hardware test split (never seen by DEM calibration or training).
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch

from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import prepare_real_datasets, evaluate_dataset
from dem_dataset import DEMDataset
from build_dem_from_detection_events import build_dem_from_detection_events
from stim_alignment import build_stim_alignment, ibm_detections_to_stim_order
from utils import TrainingLogger


D, T = 3, 10
TRAIN_JOBS = [
    "jobs/dist3/job_d3_T10_shots100000_d7b87q15a5qc73dn58rg_.json",
    "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json",
]
PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_ibm_dem"
PATIENCE = 50

args = Args(
    distance=D,
    dt=2,
    batch_size=2048,
    n_batches=400,
    n_epochs=200,
    lr=3e-4,
    min_lr=1e-6,
)

# --- Split real IBM shots 85/0/15 (only train split is used for DEM calibration)
sc = SurfaceCodeCircuit(distance=D, T=T)
real_train, real_val, real_test = prepare_real_datasets(
    sc, TRAIN_JOBS, dt=args.dt, k=args.k,
    batch_size=args.batch_size, device=args.device,
)

print(f"Real shots — train: {len(real_train.logical_flips)}, "
      f"val: {len(real_val.logical_flips)}, test: {len(real_test.logical_flips)}")

# --- Calibrate DEM from the 100% train split
alignment = build_stim_alignment(sc, rounds=T)
det_stim = ibm_detections_to_stim_order(
    real_train.detections, alignment.ibm_middle_order, alignment.ibm_z_order,
)
print(f"Calibrating DEM from {len(det_stim)} train shots...")
dem = build_dem_from_detection_events(alignment.circuit, det_stim)

# --- Three independent DEMDataset instances (Monte Carlo train/val/test)
dem_train = DEMDataset(args, dem=dem, rounds=T, circuit=alignment.circuit,
                       detector_is_z=alignment.detector_is_z)
dem_val = DEMDataset(args, dem=dem, rounds=T, circuit=alignment.circuit,
                     detector_is_z=alignment.detector_is_z)
dem_test = DEMDataset(args, dem=dem, rounds=T, circuit=alignment.circuit,
                      detector_is_z=alignment.detector_is_z)

# --- Load pretrained model
model = GRUDecoder(args)
ckpt = torch.load(PRETRAINED, weights_only=False, map_location=args.device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.to(args.device)

# --- Train with val-based early stopping
logger = TrainingLogger(logfile="finetune_dem.log", statsfile="finetune_dem")
model.train_model(
    dataset=dem_train,
    val_dataset=None,
    n_val_batches=30,
    patience=None,
    save=SAVE_NAME,
    logger=logger,
)

# --- Evaluation
def _report(name, metrics, T):
    acc = metrics["acc"]
    lfr_round = 1.0 - acc ** (1.0 / T) if acc > 0 else 1.0
    print(f"\n{name}:")
    print(f"  acc      = {acc:.4f}  (c0={metrics['acc_0']:.4f}, c1={metrics['acc_1']:.4f})")
    print(f"  shots    = {metrics['n_0'] + metrics['n_1']}  "
          f"(class-0: {metrics['n_0']}, class-1: {metrics['n_1']})")
    print(f"  LFR/round = {lfr_round:.4f}  (1 - acc^(1/T) with T={T})")

dem_test = evaluate_dataset(model, dem_test, n_batches=40)
real_test = evaluate_dataset(model, real_test, n_batches=40)
_report("DEM test", dem_test, T)
_report("Real test", real_test, T)
