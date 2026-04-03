import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import IBMJobDecoder

D, T = 3, 10
JOB = "fine_tune_jobs/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json"
PRETRAINED = "models/distance3_ibm.pt"

args = Args(
    distance=D,
    dt=5,
    batch_size=512,
    n_batches=10,
    n_epochs=200,
    embedding_features=[5, 32, 64, 128, 256],
    hidden_size=128,
    n_layers=4,
    lr=1e-4,
)

sc = SurfaceCodeCircuit(distance=D, T=T)
dataset = IBMJobDecoder(sc, job_path=JOB, dt=args.dt, k=args.k, batch_size=args.batch_size)

model = GRUDecoder(args)
ckpt = torch.load(PRETRAINED, weights_only=False)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.train_model(dataset=dataset, save="distance3_ibm")
