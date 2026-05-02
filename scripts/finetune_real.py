import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import prepare_real_datasets, evaluate_dataset
from utils import TrainingLogger, print_test_result
from mwpm_decoder import evaluate_mwpm_split


D, T = 3, 10
TRAIN_JOBS = [
    "jobs/dist3/job_d3_T10_shots100000_d7b87q15a5qc73dn58rg_.json",
    "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json",
]

PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_ibm_real_t10"
PATIENCE = 50

args = Args(
    distance=D,
    dt=2,
    batch_size=64,
    n_batches=2500,
    n_epochs=200,
    lr=1.5e-5,
    min_lr=1e-6,
)

# Load pretrained model
model = GRUDecoder(args)
ckpt = torch.load(PRETRAINED, weights_only=False, map_location=args.device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.to(args.device)

# Build train/val/test
sc = SurfaceCodeCircuit(distance=D, T=T)
real_train, real_val, real_test = prepare_real_datasets(
    sc, TRAIN_JOBS,
    dt=args.dt, k=args.k, batch_size=args.batch_size, device=args.device,
)

logger = TrainingLogger(logfile=f"{SAVE_NAME}.log", statsfile="finetune_real")
model.train_model(
    dataset=real_train,
    val_dataset=real_val,
    n_val_batches=200,
    patience=PATIENCE,
    save=SAVE_NAME,
    logger=logger,
)

# Evaluation on held-out real test
real_test_m = evaluate_dataset(model, real_test, all_shots=True)
print_test_result(real_test_m, T)

mwpm_test_m = evaluate_mwpm_split(sc, [real_train, real_val], real_test)
print_test_result(mwpm_test_m, T, label="MWPM real test")
