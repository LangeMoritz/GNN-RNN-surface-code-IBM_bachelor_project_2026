import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import IBMJobDecoder, split_ibm_job, evaluate_dataset
from utils import TrainingLogger


D, T = 3, 10
TRAIN_JOB = "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json"
TEST_JOB = "jobs/dist3/job_d3_T10_shots10000_d7b87q15a5qc73dn58rg_.json"
PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_ibm_real"
PATIENCE = 40

# batch_size * n_batches = 8192 * 11 = 90_112, covers the 85k train split once per epoch
args = Args(
    distance=D,
    dt=2,
    batch_size=8192,
    n_batches=11,
    n_epochs=140,
    lr=5e-5,
    min_lr=1e-6,
)

# --- Load pretrained model
model = GRUDecoder(args)
ckpt = torch.load(PRETRAINED, weights_only=False, map_location=args.device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.to(args.device)

sc = SurfaceCodeCircuit(distance=D, T=T)
# Train job: 85/15 train/val split (test comes from the separate TEST_JOB).
real_train, real_val = split_ibm_job(
    sc, TRAIN_JOB, ratios=[0.85, 0.15], seed=42,
    dt=args.dt, k=args.k, batch_size=args.batch_size, device=args.device,
)
real_test = IBMJobDecoder(
    sc, job_path=TEST_JOB, dt=args.dt, k=args.k,
    batch_size=args.batch_size, device=args.device,
)
real_test._load_job_data()

print(f"TRAIN_JOB: {TRAIN_JOB}")
print(f"TEST_JOB:  {TEST_JOB}")
print(f"Real shots — train: {len(real_train.logical_flips)}, "
      f"val: {len(real_val.logical_flips)}, test: {len(real_test.logical_flips)}")

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
