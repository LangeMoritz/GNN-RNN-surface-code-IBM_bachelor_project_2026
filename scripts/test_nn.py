import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import evaluate_dataset, prepare_real_datasets
from mwpm_decoder import evaluate_mwpm_split
from utils import print_test_result



# d3 T10, 100k-shot jobs
# TRAIN_JOBS = [
#     "jobs/dist3/job_d3_T10_shots100000_d7b87q15a5qc73dn58rg_.json",
#     "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json",
# ]

# d3 T20 jobs
# TRAIN_JOBS = [
#     "jobs/dist3/job_d3_T20_shots100000_d7l210a8ui0s73b5s25g.json",
#     "jobs/dist3/d3_T20_shots50000_d7p0uhu0b9ts73cj0e80.json",
#     "jobs/dist3/job_d3_T20_shots50000_d7fmgem2cugc739qov6g.json"
# ]

# d5 T10, 100k-shot jobs
# TRAIN_JOBS = [
#     "jobs/dist5/job_d5_T10_shots100000_d7jman1s7cos73ek3djg.json",
#     "jobs/dist5/d5_T10_shots100000_d7oben62jamc73bpfv00.json",
# ]

# d5 T20, 200k-shot jobs
TRAIN_JOBS = [
    "jobs/dist5/d5_T20_shots150000_d7s7584t738s73cgb1u0.json",
    "jobs/dist5/job_d5_T20_shots50000_d7fmn4l6agrc738ispv0.json"
]

D, T = 5, 20
MODEL_PATH = f"models/distance{D}.pt"

if __name__ == "__main__":
    args = Args(distance=D, dt=2)
    sc = SurfaceCodeCircuit(distance=D, T=T)

    # --- GNN-RNN ---
    model = GRUDecoder(args)
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=args.device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(args.device)

    real_train, real_val, real_test = prepare_real_datasets(
        sc, TRAIN_JOBS,
        dt=args.dt, k=args.k, batch_size=args.batch_size, device=args.device,
    )
    metrics = evaluate_dataset(model, real_test, all_shots=True)
    print_test_result(metrics, T, label=f"GNN-RNN (d={D}, T={T})")

    # --- MWPM ---
    metrics = evaluate_mwpm_split(
        sc, [real_train, real_val], real_test,
    )
    print_test_result(metrics, T, label="MWPM real test")
