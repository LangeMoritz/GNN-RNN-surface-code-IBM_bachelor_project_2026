import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from args import Args
from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import prepare_real_datasets
from mwpm_decoder import evaluate_mwpm_split
from utils import print_test_result


D, T = 3, 10
TRAIN_JOBS = [
    "jobs/dist3/job_d3_T10_shots100000_d7b87q15a5qc73dn58rg_.json",
    "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json",
]

args = Args(distance=D, dt=2, batch_size=64)
sc = SurfaceCodeCircuit(distance=D, T=T)
real_train, real_val, real_test = prepare_real_datasets(
    sc, TRAIN_JOBS,
    dt=args.dt, k=args.k,
    batch_size=args.batch_size, device=args.device,
)

metrics = evaluate_mwpm_split(sc, [real_train, real_val], real_test)
print_test_result(metrics, T, label=f"MWPM real test (d={D}, T={T})")
