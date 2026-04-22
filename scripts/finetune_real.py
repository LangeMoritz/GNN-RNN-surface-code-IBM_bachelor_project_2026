import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import IBMJobDecoder, split_ibm_job, evaluate_dataset
from utils import TrainingLogger


D, T = 3, 10
TRAIN_JOB_A = "jobs/dist3/job_d3_T10_shots100000_d7b87q15a5qc73dn58rg_.json"
TRAIN_JOB_B = "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json"
TEST_JOB = "jobs/dist3/job_d7767p52b89c73d479pg_d3_T10_shots10000.json"
PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_ibm_real"
PATIENCE = 30

args = Args(
    distance=D,
    dt=2,
    batch_size=256,
    n_batches=400,
    n_epochs=200,
    lr=3e-5,
    min_lr=1e-6,
)

# --- Load pretrained model
model = GRUDecoder(args)
ckpt = torch.load(PRETRAINED, weights_only=False, map_location=args.device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.to(args.device)


def _concat_ibm_decoders(sc, datasets):
    merged = IBMJobDecoder(
        sc,
        job_path=datasets[0].job_path,
        dt=args.dt,
        k=args.k,
        batch_size=args.batch_size,
        device=args.device,
    )
    merged.detections = np.concatenate([ds.detections for ds in datasets], axis=0)
    merged.logical_flips = np.concatenate([ds.logical_flips for ds in datasets], axis=0)
    return merged


sc = SurfaceCodeCircuit(distance=D, T=T)
train_a, val_a = split_ibm_job(
    sc, TRAIN_JOB_A, ratios=[0.90, 0.10], seed=42,
    dt=args.dt, k=args.k, batch_size=args.batch_size, device=args.device,
)
train_b, val_b = split_ibm_job(
    sc, TRAIN_JOB_B, ratios=[0.90, 0.10], seed=43,
    dt=args.dt, k=args.k, batch_size=args.batch_size, device=args.device,
)

real_test = IBMJobDecoder(
    sc, job_path=TEST_JOB, dt=args.dt, k=args.k,
    batch_size=args.batch_size, device=args.device,
)
real_test._load_job_data()

real_train = _concat_ibm_decoders(sc, [train_a, train_b])
real_val = _concat_ibm_decoders(sc, [val_a, val_b])

print(f"TRAIN_JOB_A: {TRAIN_JOB_A}")
print(f"TRAIN_JOB_B: {TRAIN_JOB_B}")
print(f"TEST_JOB:    {TEST_JOB}")
print(f"Real shots — train: {len(real_train.logical_flips)}, "
    f"val: {len(real_val.logical_flips)}, "
      f"test: {len(real_test.logical_flips)}")

logger = TrainingLogger(statsfile="finetune_real")
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
print(f" acc = {acc:.4f}  (c0={real_test_m['acc_0']:.4f}, c1={real_test_m['acc_1']:.4f})")
print(f" shots = {real_test_m['n_0'] + real_test_m['n_1']}  "
      f"(class-0: {real_test_m['n_0']}, class-1: {real_test_m['n_1']})")
print(f" LFR/round = {lfr_round:.4f}  (1 - acc^(1/T) with T={T})")
