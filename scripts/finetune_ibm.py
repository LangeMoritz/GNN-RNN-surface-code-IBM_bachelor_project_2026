import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import IBMJobDecoder
from utils import TrainingLogger

D, T = 3, 10
JOB = "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json"
PRETRAINED = "models/distance3_ibm_test.pt"

args = Args(
    distance=D,
    dt=2,
    batch_size=512,
    n_batches=10,
    n_epochs=200,
    lr=1e-5,
    min_lr=1e-6,
)

# --- Load pretrained model ---
model = GRUDecoder(args)
ckpt = torch.load(PRETRAINED, weights_only=False, map_location=args.device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.to(args.device)

# --- Load IBM data and split 80/20 ---
sc = SurfaceCodeCircuit(distance=D, T=T)
all_data = IBMJobDecoder(
    sc, job_path=JOB, dt=args.dt, k=args.k,
    batch_size=args.batch_size, device=args.device,
)
all_data._load_job_data()
n_total = len(all_data.logical_flips)
n_test = int(n_total * 0.2)
perm = np.random.RandomState(42).permutation(n_total)
test_indices = perm[:n_test]
train_indices = perm[n_test:]

# --- Train dataset (80%) ---
train_dataset = IBMJobDecoder(
    sc, job_path=JOB, dt=args.dt, k=args.k,
    batch_size=args.batch_size, device=args.device,
)
train_dataset._load_job_data()
train_dataset.detections = train_dataset.detections[train_indices]
train_dataset.logical_flips = train_dataset.logical_flips[train_indices]

# --- Test dataset (20%) ---
test_dataset = IBMJobDecoder(
    sc, job_path=JOB, dt=args.dt, k=args.k,
    batch_size=args.batch_size, device=args.device,
)
test_dataset._load_job_data()
test_dataset.detections = test_dataset.detections[test_indices]
test_dataset.logical_flips = test_dataset.logical_flips[test_indices]

print(f"Train: {len(train_indices)} | Test: {n_test}")

# --- Train ---
logger = TrainingLogger(logfile="finetune_ibm.log", statsfile="finetune_ibm")
model.train_model(dataset=train_dataset, save="distance3_ibm", logger=logger)

# --- Test ---
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for _ in range(10):
        x, edge_index, labels, label_map, edge_attr, flips = test_dataset.generate_batch()
        out = model.forward(x, edge_index, edge_attr, labels, label_map)
        predicted = torch.round(out).squeeze()
        correct += (predicted == flips.squeeze()).sum().item()
        total += flips.numel()
print(f"Test accuracy: {correct / total:.4f}")
