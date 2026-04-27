import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import prepare_real_datasets, evaluate_dataset
from utils import TrainingLogger, print_test_result


D, T = 3, 20

TRAIN_JOBS = [
    "jobs/dist3/job_d3_T20_shots50000_d7fmgem2cugc739qov6g.json",
]

PRETRAINED = f"models/distance{D}_ibm_real.pt"
SAVE_NAME = f"distance{D}_ibm_real_t10t20"
PATIENCE = 40

args = Args(
    distance=D,
    dt=2,
    batch_size=256,
    n_batches=200,
    n_epochs=200,
    lr=1e-4,
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

logger = TrainingLogger(statsfile="finetune_real")
model.train_model(
    dataset=real_train,
    val_dataset=real_val,
    n_val_batches=50,
    patience=PATIENCE,
    save=SAVE_NAME,
    logger=logger,
)

# Evaluation on held-out real test
real_test_m = evaluate_dataset(model, real_test, n_batches=40)
print_test_result(real_test_m, T)
