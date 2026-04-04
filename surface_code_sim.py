import json
import os
from dotenv import load_dotenv

import torch
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.exceptions import QiskitError

from surface_code_miami import SurfaceCodeCircuit
from ibm_decoder import IBMJobDecoder
from gru_decoder import GRUDecoder
from args import Args


def get_runtime_service():
    load_dotenv()
    return QiskitRuntimeService(
        token=os.getenv("IBM_KEY"),
        instance="Surface Codes - Bachelor Thesis 2",
    )


def build_simple_noise_model(p1=0.0, p2=0.0, p_meas=0.0):
    """
    Enkel brusmodell.

    Parametrar
    ----------
    p1 : float
        1-qubit depolarizing noise.
    p2 : float
        2-qubit depolarizing noise.
    p_meas : float
        Symmetriskt readout error.
    """
    noise_model = NoiseModel()

    if p1 > 0:
        err1 = depolarizing_error(p1, 1)
        # Lägg på vanliga 1-qubit-grindar som kan förekomma efter transpilation
        for gate in ["h", "x", "sx", "id", "rz"]:
            try:
                noise_model.add_all_qubit_quantum_error(err1, [gate])
            except Exception:
                pass

    if p2 > 0:
        err2 = depolarizing_error(p2, 2)
        for gate in ["cx", "ecr", "cz"]:
            try:
                noise_model.add_all_qubit_quantum_error(err2, [gate])
            except Exception:
                pass

    if p_meas > 0:
        readout = ReadoutError([
            [1 - p_meas, p_meas],
            [p_meas, 1 - p_meas],
        ])
        noise_model.add_all_qubit_readout_error(readout)

    return noise_model


def simulate_surface_code(
    distance: int,
    T: int,
    shots: int,
    backend_name: str = "ibm_miami",
    simulation_mode: str = "ideal",
    p1: float = 0.0,
    p2: float = 0.0,
    p_meas: float = 0.0,
):
    """
    simulation_mode:
        - "ideal"         : obrusig simulering
        - "backend_noise" : IBM-lik noise från AerSimulator.from_backend(...)
        - "custom_noise"  : enkel egen brusmodell
    """
    service = get_runtime_service()
    backend = service.backend(backend_name)

    sc = SurfaceCodeCircuit(distance=distance, T=T)

    transpiled = transpile(
        sc.circuit,
        backend=backend,
        initial_layout=sc.make_layout(),
        routing_method="none",
        optimization_level=1,
        seed_transpiler=42,
    )

    if simulation_mode == "ideal":
        # Stabilizer-metoden är mycket lättare för denna typ av krets
        simulator = AerSimulator(method="stabilizer")
    elif simulation_mode == "backend_noise":
        simulator = AerSimulator.from_backend(backend)
    elif simulation_mode == "custom_noise":
        noise_model = build_simple_noise_model(p1=p1, p2=p2, p_meas=p_meas)
        simulator = AerSimulator(noise_model=noise_model)
    else:
        raise ValueError(
            "simulation_mode måste vara 'ideal', 'backend_noise' eller 'custom_noise'"
        )

    try:
        result = simulator.run(transpiled, shots=shots).result()
        counts = result.get_counts()

    except QiskitError as e:
        msg = str(e)
        if "Insufficient memory" in msg:
            print("\nSimuleringen avbröts: minnet tog slut.")
            print("Testa något av följande:")
            print("- minska DISTANCE, t.ex. till 3")
            print("- använd simulation_mode='ideal'")
            print("- använd simulation_mode='custom_noise' med liten brusnivå")
            print("- undvik full backend_noise för stora kretsar")
            return None
        raise

    clean_counts = {}
    for bitstring, count in counts.items():
        clean_bitstring = bitstring.replace(" ", "")
        clean_counts[clean_bitstring] = clean_counts.get(clean_bitstring, 0) + count

    output_path = os.path.join(
        "sim_jobs",
        f"{backend_name}_surface_code_d{distance}_T{T}_shots{shots}_{simulation_mode}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "counts": clean_counts
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Simulation saved to {output_path}")
    print("Example raw bitstring:    ", next(iter(counts)))
    print("Example cleaned bitstring:", next(iter(clean_counts)))

    return output_path


def evaluate_simulated_job(job_path: str, distance: int, T: int, model_path: str):
    """
    Kör samma decoder-pipeline på den simulerade datan.
    """
    args = Args(
        distance=distance,
        dt=2,
        embedding_features=[5, 32, 64, 128, 256],
        hidden_size=128,
        n_layers=4,
    )

    model = GRUDecoder(args)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    sc = SurfaceCodeCircuit(distance=distance, T=T)

    dataset = IBMJobDecoder(
        sc,
        job_path=job_path,
        simulator=True,
        dt=args.dt,
        k=args.k,
    )

    try:
        x, edge_index, labels, label_map, edge_attr, flips = dataset.generate_batch()
    except ValueError as e:
        if "need at least one array to concatenate" in str(e):
            print("\nInga detection events hittades.")
            print("Detta motsvarar perfekt/noiseless beteende för den här batchen.")
            print("Accuracy tolkas därför som 1.0000")
            return 1.0
        raise

    with torch.no_grad():
        predictions = model.forward(x, edge_index, edge_attr, labels, label_map)

    predicted_flips = torch.round(predictions).int()
    accuracy = (predicted_flips.squeeze() == flips.squeeze()).float().mean()

    print(f"GNN-RNN accuracy on simulated data: {accuracy:.4f}")
    return accuracy.item()


if __name__ == "__main__":
    DISTANCE = 3
    T = 10
    SHOTS = 1000
    BACKEND = "ibm_miami"

    # Välj ett av: "ideal", "backend_noise", "custom_noise"
    SIMULATION_MODE = "custom_noise"

    # Egen brusmodell används bara om SIMULATION_MODE = "custom_noise"
    P1 = 0.0001    # 1-qubit noise
    P2 = 0.000     # 2-qubit noise
    P_MEAS = 0.00 # readout noise

    sim_path = simulate_surface_code(
        distance=DISTANCE,
        T=T,
        shots=SHOTS,
        backend_name=BACKEND,
        simulation_mode=SIMULATION_MODE,
        p1=P1,
        p2=P2,
        p_meas=P_MEAS,
    )

    if sim_path is not None:
        evaluate_simulated_job(
            job_path=sim_path,
            distance=DISTANCE,
            T=T,
            model_path="GNN-RNN-surface-code-IBM_bachelor_project_2026/models/distance5.pt",
        )