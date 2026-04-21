
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch

from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import split_ibm_job, evaluate_dataset
from utils import TrainingLogger


D, T = 3, 10
JOBS = [
    "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json",
    "jobs/dist3/job_d7767p52b89c73d479pg_d3_T10_shots10000.json",
]
PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_real_full"

args = Args(
    distance=D,
    dt=2,
    batch_size=1024,
    n_batches=32,
    n_epochs=400,
    lr=1e-4,
    min_lr=1e-6,
)

model = GRUDecoder(args)
ckpt = torch.load(PRETRAINED, weights_only=False, map_location=args.device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.to(args.device)

sc = SurfaceCodeCircuit(distance=D, T=T)
final_test = None

for stage_idx, job in enumerate(JOBS, start=1):
    is_last = stage_idx == len(JOBS)
    ratios = [0.80, 0.0, 0.20] if is_last else [1.0, 0.0, 0.0]

    real_train, _, real_test = split_ibm_job(
        sc, job, ratios=ratios, seed=42,
        dt=args.dt, k=args.k, batch_size=args.batch_size, device=args.device,
    )
    if is_last:
        final_test = real_test

    print(f"\n=== Stage {stage_idx}/{len(JOBS)} ===")
    print(f"JOB: {job}")
    print(f"Real shots — train: {len(real_train.logical_flips)}, "
          f"test: {len(real_test.logical_flips)}")

    logger = TrainingLogger(
        logfile=f"finetune_real_full_stage{stage_idx}.log",
        statsfile=f"finetune_real_full_stage{stage_idx}",
    )
    stage_save = f"{SAVE_NAME}_stage{stage_idx}" if not is_last else SAVE_NAME
    model.train_model(
        dataset=real_train,
        save=stage_save,
        logger=logger,
    )

real_test_m = evaluate_dataset(model, final_test, n_batches=40)
acc = real_test_m["acc"]
lfr_round = 1.0 - acc ** (1.0 / T) if acc > 0 else 1.0
print(f"\nFinal-stage real test (20% of last job):")
print(f"  acc       = {acc:.4f}  (c0={real_test_m['acc_0']:.4f}, c1={real_test_m['acc_1']:.4f})")
print(f"  shots     = {real_test_m['n_0'] + real_test_m['n_1']}  "
      f"(class-0: {real_test_m['n_0']}, class-1: {real_test_m['n_1']})")
print(f"  LFR/round = {lfr_round:.4f}  (1 - acc^(1/T) with T={T})")
