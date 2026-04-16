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
PATIENCE = 30

args = Args(
    distance=D,
    dt=2,
    batch_size=2048,
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
    n_val_batches=30,
    patience=PATIENCE,
    save=SAVE_NAME,
    logger=logger,
)

# --- Evaluation on held-out real test
real_test_m = evaluate_dataset(model, real_test, n_batches=40)
acc = real_test_m["acc"]
lfr_round = 1.0 - acc ** (1.0 / T) if acc > 0 else 1.0
print(f"\nReal test:")
print(f"  acc       = {acc:.4f}  (c0={real_test_m['acc_0']:.4f}, c1={real_test_m['acc_1']:.4f})")
print(f"  shots     = {real_test_m['n_0'] + real_test_m['n_1']}  "
      f"(class-0: {real_test_m['n_0']}, class-1: {real_test_m['n_1']})")
print(f"  LFR/round = {lfr_round:.4f}  (1 - acc^(1/T) with T={T})")
