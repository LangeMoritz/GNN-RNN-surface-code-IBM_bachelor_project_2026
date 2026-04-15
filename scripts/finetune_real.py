"""
Fine-tune on real IBM hardware shots only, with 75/15/15 train/val/test
split and early stopping on val accuracy.

Uses the same seed (42) as the DEM and DEM→real scripts so the test split
is identical across all three regimes.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch

from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import split_ibm_job, evaluate_dataset
from utils import TrainingLogger


D, T = 3, 20
JOB = "jobs/dist3/job_d3_T20_shots50000_d7fmgem2cugc739qov6g.json"
PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_ibm_real"

args = Args(
    distance=D,
    dt=5,
    batch_size=512,
    n_batches=32,
    n_epochs=400,
    lr=1e-4,
    min_lr=1e-6,
)

# --- Split real shots 75/15/15 (same seed as the other scripts)
sc = SurfaceCodeCircuit(distance=D, T=T)
real_train, real_val, real_test = split_ibm_job(
    sc, JOB, ratios=[0.70, 0.15, 0.15], seed=42,
    dt=args.dt, k=args.k, batch_size=args.batch_size, device=args.device,
)
print(f"Real shots — train: {len(real_train.logical_flips)}, "
      f"val: {len(real_val.logical_flips)}, test: {len(real_test.logical_flips)}")

# --- Load pretrained model
model = GRUDecoder(args)
ckpt = torch.load(PRETRAINED, weights_only=False, map_location=args.device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.to(args.device)

# --- Train
logger = TrainingLogger(logfile="finetune_real.log", statsfile="finetune_real")
model.train_model(
    dataset=real_train,
    val_dataset=real_val,
    n_val_batches=10,
    patience=40,
    save=SAVE_NAME,
    logger=logger,
)

# --- Evaluation on held-out real test
real_test_acc = evaluate_dataset(model, real_test, n_batches=40)
print(f"\nReal test accuracy: {real_test_acc:.4f}")
