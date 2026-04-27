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
from utils import TrainingLogger, print_test_result


D, T = 3, 10
TRAIN_JOBS = [
    "jobs/dist3/job_d3_T10_shots100000_d7b87q15a5qc73dn58rg_.json",
    "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json",
]
PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_ibm_dem"
PATIENCE = 80

args = Args(
    distance=D,
    dt=2,
    batch_size=1024,
    n_batches=600,
    n_epochs=300,
    lr=1e-4,
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
    val_dataset=dem_val,
    n_val_batches=100,
    patience=PATIENCE,
    save=SAVE_NAME,
    logger=logger,
)

# --- Evaluation
dem_test_m = evaluate_dataset(model, dem_test, n_batches=40)
real_test_m = evaluate_dataset(model, real_test, n_batches=40)
print_test_result(dem_test_m, T, label="DEM test")
print_test_result(real_test_m, T, label="Real test")
