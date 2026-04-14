"""
Build a stim.Circuit that matches IBM's SurfaceCodeCircuit exactly.

Same boundary stabilizers, same CNOT ordering, same detector definitions.
This replaces stim.Circuit.generated() as the DEM template so that
build_dem_from_detection_events gets the correct graph structure for
IBM hardware data.

"""

import stim
import numpy as np
from surface_code_miami import SurfaceCodeCircuit, X_ORDER, Z_ORDER


def build_ibm_stim_circuit(distance, rounds, noise=1e-3, corner_qubit=5):
    """
    Build a stim.Circuit matching IBM's SurfaceCodeCircuit.

    The circuit has the same boundary stabilizers and CNOT ordering as IBM,
    with placeholder depolarizing noise (the DEM builder replaces weights
    with empirical values anyway).

    Detector ordering:
      - Round 0: Z-type only (sorted by ancilla index)
      - Rounds 1..T-1: all ancillas (in index order)
      - Final: Z-type only (sorted by ancilla index)
    """
    d = distance
    n_data = d * d
    sc = SurfaceCodeCircuit(distance=d, T=0, corner_qubit=corner_qubit)
    n_anc = len(sc.ancilla_physical)
    x_type = sc.x_type
    z_type_sorted = sorted(set(range(n_anc)) - x_type)

    # CNOT schedule: 4 steps, each a list of (control, target) pairs
    cnot_steps = [[] for _ in range(4)]
    stabilizer_neighbors = {}

    for anc_i, anc_p in enumerate(sc.ancilla_physical):
        is_x = anc_i in x_type
        order = X_ORDER if is_x else Z_ORDER
        neighbors = []
        for step, direction in enumerate(order):
            nb = anc_p + direction
            if nb in sc.data_set:
                dat_i = sc.data_idx[nb]
                neighbors.append(dat_i)
                anc_q = n_data + anc_i
                if is_x:
                    cnot_steps[step].append((anc_q, dat_i))
                else:
                    cnot_steps[step].append((dat_i, anc_q))
        stabilizer_neighbors[anc_i] = neighbors

    # Detector coordinates (centroid of neighbor data qubits)
    det_coords = {}
    for anc_i in range(n_anc):
        nbs = stabilizer_neighbors[anc_i]
        det_coords[anc_i] = (
            float(np.mean([n % d for n in nbs])),
            float(np.mean([n // d for n in nbs])),
        )

    c = stim.Circuit()

    # Qubit coordinates
    for i in range(n_data):
        c.append("QUBIT_COORDS", i, [float(i % d), float(i // d)])
    for anc_i in range(n_anc):
        cx, cy = det_coords[anc_i]
        c.append("QUBIT_COORDS", n_data + anc_i, [cx, cy])

    # Reset all
    c.append("R", list(range(n_data + n_anc)))

    anc_qubits = list(range(n_data, n_data + n_anc))
    x_anc_qubits = sorted([n_data + i for i in x_type])
    data_qubits = list(range(n_data))

    for r in range(rounds):
        c.append("TICK")

        # Data idle noise
        c.append("DEPOLARIZE1", data_qubits, noise)

        # H on X-type ancillas
        if x_anc_qubits:
            c.append("H", x_anc_qubits)

        # 4 CNOT steps
        for step in range(4):
            c.append("TICK")
            targets = []
            for ctrl, targ in cnot_steps[step]:
                targets.extend([ctrl, targ])
            if targets:
                c.append("CX", targets)
                c.append("DEPOLARIZE2", targets, noise)

        c.append("TICK")

        # H on X-type ancillas
        if x_anc_qubits:
            c.append("H", x_anc_qubits)

        # Measure + reset ancillas
        c.append("X_ERROR", anc_qubits, noise)
        c.append("MR", anc_qubits)

        # Detectors
        if r == 0:
            # Z-type only (X-type initial eigenvalue is random in Z memory)
            for anc_i in z_type_sorted:
                cx, cy = det_coords[anc_i]
                c.append("DETECTOR",
                         [stim.target_rec(-(n_anc - anc_i))],
                         [cx, cy, 0])
        else:
            # All types: compare to previous measurement
            for anc_i in range(n_anc):
                cx, cy = det_coords[anc_i]
                c.append("DETECTOR",
                         [stim.target_rec(-(n_anc - anc_i)),
                          stim.target_rec(-(2 * n_anc - anc_i))],
                         [cx, cy, r])

    # Final data measurement
    c.append("TICK")
    c.append("X_ERROR", data_qubits, noise)
    c.append("M", data_qubits)

    # Final Z-type detectors: data qubit parity vs last Z measurement
    for anc_i in z_type_sorted:
        neighbors = stabilizer_neighbors[anc_i]
        targets = [stim.target_rec(-(n_data - dat_i)) for dat_i in neighbors]
        targets.append(stim.target_rec(-(n_data + n_anc - anc_i)))
        cx, cy = det_coords[anc_i]
        c.append("DETECTOR", targets, [cx, cy, rounds])

    # Logical observable: top row parity
    logical_targets = [stim.target_rec(-(n_data - q)) for q in range(d)]
    c.append("OBSERVABLE_INCLUDE", logical_targets, 0)

    return c


def ibm_detections_to_stim(detections_3d, x_type):
    """
    Convert IBM detection events to match the custom stim circuit's detector ordering.

    IBM has shape (n_shots, T+1, n_anc) with all ancillas every round,
    but X-type in rounds 0 and T are always 0. This function drops those
    columns to match the stim circuit which only defines Z-type detectors
    in rounds 0 and T.

    Parameters
    ----------
    detections_3d : np.ndarray of shape (n_shots, T+1, n_anc)
        IBM detection events from IBMJobDecoder._load_job_data().
    x_type : set of int
        Indices of X-type ancillas.

    Returns
    -------
    np.ndarray of shape (n_shots, num_stim_detectors), dtype bool
    """
    n_anc = detections_3d.shape[2]
    z_cols = sorted(set(range(n_anc)) - x_type)

    parts = []
    # Round 0: Z-type only
    parts.append(detections_3d[:, 0, z_cols])
    # Rounds 1..T-1: all ancillas
    for r in range(1, detections_3d.shape[1] - 1):
        parts.append(detections_3d[:, r, :])
    # Final round: Z-type only
    parts.append(detections_3d[:, -1, z_cols])

    return np.hstack(parts).astype(bool)
