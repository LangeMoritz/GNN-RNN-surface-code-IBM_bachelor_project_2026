"""
Two-phase fine-tuning:
  Phase A — train on DEM-sampled data with real val set (early stop).
  Phase B — continue training on real hardware shots with real val set.
"""
import sys, os
import atexit
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import prepare_real_datasets, evaluate_dataset
from dem_dataset import DEMDataset
from build_dem_from_detection_events import build_dem_from_detection_events
from stim_alignment import build_stim_alignment, ibm_detections_to_stim_order
from utils import TrainingLogger, print_test_result


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        self.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


D, T = 3, 10
TRAIN_JOBS = [
    "jobs/dist3/job_d3_T10_shots100000_d7b87q15a5qc73dn58rg_.json",
    "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json",
]

PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_ibm_dem_real"
PATIENCE_A = 40
PATIENCE_B = 60
SAVE_CONSOLE_LOG = False
CONSOLE_LOG_PATH = f"jobs/logs/{SAVE_NAME}_console.log"

if SAVE_CONSOLE_LOG:
    os.makedirs(os.path.dirname(CONSOLE_LOG_PATH), exist_ok=True)
    _console_log = open(CONSOLE_LOG_PATH, "a", buffering=1)
    atexit.register(_console_log.close)
    sys.stdout = Tee(sys.stdout, _console_log)
    sys.stderr = Tee(sys.stderr, _console_log)
    print(f"Saving console output to {CONSOLE_LOG_PATH}", flush=True)


# Phase A (DEM-sampled)
args_dem = Args(
    distance=D,
    dt=2,
    batch_size=256,
    n_batches=195,
    n_epochs=300,
    lr=3e-4, # testa 5e-6
    min_lr=1e-6,
)
# Phase B (real samples)
args_real = Args(
    distance=D,
    dt=2,
    batch_size=64,
    n_batches=1600,
    n_epochs=300,
    lr=2e-5, # 2e-5 if overfitts, 1e-4 if underfits
    min_lr=1e-6,
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
    detector_is_z=alignment.detector_is_z,
)

# Model + Phase A
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

# Phase B: continue on real hardware shots
# Swap model args so the new optimizer uses the real-phase LR schedule.
model.args = args_real

print("\n=== Phase B: real-data fine-tune ===")
logger_b = TrainingLogger(statsfile="finetune_dem_real_phaseB")
model.train_model(
    dataset=real_train, val_dataset=real_val,
    n_val_batches=30, patience=PATIENCE_B,
    save=SAVE_NAME, logger=logger_b,
)

# Final evaluation on held-out real test
real_test_m = evaluate_dataset(model, real_test, n_batches=40)
print_test_result(real_test_m, T)
