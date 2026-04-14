from dataclasses import dataclass

import numpy as np
import stim

from surface_code_miami import SurfaceCodeCircuit


@dataclass
class StimAlignment:
    circuit: stim.Circuit
    rounds: int
    ibm_middle_order: np.ndarray
    ibm_z_order: np.ndarray
    ibm_ancilla_xy: np.ndarray
    detector_is_z: np.ndarray


ALIGNMENT_TABLES = {
    3: {
        "ibm_middle_order": [0, 3, 2, 1, 6, 5, 4, 7],
        "ibm_z_order": [6, 3, 4, 1],
        "ibm_ancilla_xy": [
            [1.0, 0.0], [3.0, 1.0], [2.0, 1.0], [1.0, 1.0],
            [2.0, 2.0], [1.0, 2.0], [0.0, 2.0], [2.0, 3.0],
        ],
    },
    5: {
        "ibm_middle_order": [
            1, 0, 6, 5, 4, 3, 2, 11, 10, 9, 8, 7,
            16, 15, 14, 13, 12, 21, 20, 19, 18, 17, 23, 22,
        ],
        "ibm_z_order": [11, 21, 6, 16, 9, 19, 4, 14, 7, 17, 2, 12],
        "ibm_ancilla_xy": [
            [3.0, 0.0], [1.0, 0.0], [5.0, 1.0], [4.0, 1.0], [3.0, 1.0], [2.0, 1.0],
            [1.0, 1.0], [4.0, 2.0], [3.0, 2.0], [2.0, 2.0], [1.0, 2.0], [0.0, 2.0],
            [5.0, 3.0], [4.0, 3.0], [3.0, 3.0], [2.0, 3.0], [1.0, 3.0], [4.0, 4.0],
            [3.0, 4.0], [2.0, 4.0], [1.0, 4.0], [0.0, 4.0], [4.0, 5.0], [2.0, 5.0],
        ],
    },
}


def build_stim_alignment(
    sc: SurfaceCodeCircuit,
    rounds: int,
    noise: float = 1e-3,
) -> StimAlignment:
    distance = sc.distance
    table = ALIGNMENT_TABLES[distance]
    ibm_middle_order = np.array(table["ibm_middle_order"], dtype=np.int64)
    ibm_z_order = np.array(table["ibm_z_order"], dtype=np.int64)
    ibm_ancilla_xy = np.array(table["ibm_ancilla_xy"], dtype=np.float32)
    n_z = len(ibm_z_order)

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
    )

    detector_is_z = []
    detector_is_z.extend([1.0] * n_z)
    for _ in range(1, rounds):
        for anc_i in ibm_middle_order:
            detector_is_z.append(0.0 if anc_i in sc.x_type else 1.0)
    detector_is_z.extend([1.0] * n_z)
    detector_is_z = np.array(detector_is_z, dtype=np.float32)

    return StimAlignment(
        circuit=circuit,
        rounds=rounds,
        ibm_middle_order=ibm_middle_order,
        ibm_z_order=ibm_z_order,
        ibm_ancilla_xy=ibm_ancilla_xy,
        detector_is_z=detector_is_z,
    )


def ibm_detections_to_stim_order(
    detections_3d: np.ndarray,
    ibm_middle_order: np.ndarray,
    ibm_z_order: np.ndarray,
) -> np.ndarray:
    parts = [detections_3d[:, 0, ibm_z_order]]
    for r in range(1, detections_3d.shape[1] - 1):
        parts.append(detections_3d[:, r, ibm_middle_order])
    parts.append(detections_3d[:, -1, ibm_z_order])

    return np.hstack(parts).astype(bool)
