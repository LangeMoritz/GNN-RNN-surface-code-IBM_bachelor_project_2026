import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import IBMJobDecoder
from dem_dataset import DEMDataset
from build_dem_from_detection_events import build_dem_from_detection_events
from stim_alignment import (
    build_stim_alignment,
    ibm_detections_to_stim_order,
)
from utils import TrainingLogger


def evaluate(model, dataset, n_batches):
    """Evaluate model accuracy on a dataset (real IBM shots)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, edge_index, labels, label_map, edge_attr, flips = dataset.generate_batch()
            out = model.forward(x, edge_index, edge_attr, labels, label_map)
            predicted = torch.round(out).squeeze()
            correct += (predicted == flips.squeeze()).sum().item()
            total += flips.numel()
    model.train()
    return correct / total if total > 0 else 0.0


D, T = 3, 10
JOB = "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json"
PRETRAINED = "models/distance3_ibm_test.pt"

args = Args(
    distance=D,
    dt=2,
    batch_size=512,
    n_batches=10,
    n_epochs=200,
    lr=1e-3,
    min_lr=1e-4,
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

# --- Build DEM from 80% train split ---
train_detections = all_data.detections[train_indices]
alignment = build_stim_alignment(sc, rounds=T)
circuit = alignment.circuit
det_stim = ibm_detections_to_stim_order(
    train_detections,
    alignment.ibm_middle_order,
    alignment.ibm_z_order,
)
print(f"Building DEM from {len(train_detections)} shots...")
dem = build_dem_from_detection_events(circuit, det_stim)
train_dataset = DEMDataset(
    args,
    dem=dem,
    rounds=T,
    circuit=circuit,
    detector_is_z=alignment.detector_is_z,
)

# --- Test dataset (20%) ---
test_dataset = IBMJobDecoder(
    sc, job_path=JOB, dt=args.dt, k=args.k,
    batch_size=args.batch_size, device=args.device,
)
test_dataset._load_job_data()
test_dataset.detections = test_dataset.detections[test_indices]
test_dataset.logical_flips = test_dataset.logical_flips[test_indices]

print(f"Train: {len(train_indices)} shots (DEM) | Test: {n_test}")

# --- Train ---
logger = TrainingLogger(logfile="finetune_dem.log", statsfile="finetune_dem")
model.train_model(dataset=train_dataset, save=f"distance{D}_ibm_test", logger=logger)

# --- Test on held-out set ---
test_acc = evaluate(model, test_dataset, n_batches=10)
print(f"Test accuracy: {test_acc:.4f}")
