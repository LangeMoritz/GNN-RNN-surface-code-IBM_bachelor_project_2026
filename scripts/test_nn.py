import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from args import Args
from gru_decoder import GRUDecoder
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import IBMJobDecoder, evaluate_dataset
from mwpm_decoder import MWPMDecoder
from utils import print_test_result, lfr_per_round


D, T = 3, 20
TEST_JOB = "jobs/d3_T20_shots500_d7p0kjj9ak2c739r4nt0.json"
MODEL_PATH = f"models/distance{D}.pt"
PIJ_THRESHOLD = 0.044  # d=3, 0 for d=5


if __name__ == "__main__":
    args = Args(distance=D, dt=2)
    sc = SurfaceCodeCircuit(distance=D, T=T)

    # --- GNN-RNN ---
    model = GRUDecoder(args)
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=args.device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(args.device)

    dataset = IBMJobDecoder(
        sc, job_path=TEST_JOB, dt=args.dt, k=args.k,
        batch_size=args.batch_size, device=args.device,
    )
    metrics = evaluate_dataset(model, dataset, all_shots=True)
    print_test_result(metrics, T, label=f"GNN-RNN (d={D}, T={T})")

    # --- MWPM ---
    print("\nMWPM:")
    p_l, _ = MWPMDecoder(sc, job_path=TEST_JOB, pij_threshold=PIJ_THRESHOLD).decode()
    print(f" LFR/round = {lfr_per_round(1 - p_l, T):.4f}")
