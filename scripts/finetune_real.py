import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import prepare_real_datasets, evaluate_dataset
from utils import TrainingLogger, print_test_result
from mwpm_decoder import evaluate_mwpm_split


D, T = 5, 10
TRAIN_JOBS = [
    "jobs/dist5/job_d5_T10_shots100000_d7jman1s7cos73ek3djg.json",
    "jobs/dist5/d5_T10_shots100000_d7oben62jamc73bpfv00.json",
]

PRETRAINED = f"models/distance{D}.pt"
SAVE_NAME = f"distance{D}_t10_real"
PATIENCE = 60

args = Args(
    distance=D,
    dt=2,
    batch_size=64,
    n_batches=2500,
    n_epochs=200,
    lr=1e-5,
    min_lr=5e-7,
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

logger = TrainingLogger(logfile=f"{SAVE_NAME}.log", statsfile="finetune_real_t10")
model.train_model(
    dataset=real_train,
    val_dataset=real_val,
    n_val_batches=500,
    patience=PATIENCE,
    save=SAVE_NAME,
    logger=logger,
)

# Evaluation on held-out real test
real_test_m = evaluate_dataset(model, real_test, all_shots=True)
print_test_result(real_test_m, T)

mwpm_test_m = evaluate_mwpm_split(sc, [real_train, real_val], real_test)
print_test_result(mwpm_test_m, T, label="MWPM real test")
