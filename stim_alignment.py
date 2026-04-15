"""
Alignment between our IBM-chip surface code layout and stim's
`surface_code:rotated_memory_z` layout.

Everything is derived at runtime from (a) the IBM chip positions stored in
`SurfaceCodeCircuit` and (b) the detector coordinates reported by the stim
circuit.  Nothing is hard-coded per distance, so this works for any distance
supported by `CHIP_MAP`.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import stim

from surface_code_miami import SurfaceCodeCircuit


@dataclass
class StimAlignment:
    circuit: stim.Circuit
    rounds: int
    ibm_middle_order: np.ndarray   # IBM anc indices in stim middle-round order
    ibm_z_order: np.ndarray        # IBM anc indices in stim first/last-round order
    ibm_ancilla_xy: np.ndarray     # (num_anc, 2) stim (x, y) halved, IBM-indexed
    detector_is_z: np.ndarray      # (num_detectors,) 1.0 for Z, 0.0 for X


def _ibm_ancilla_xy(sc: SurfaceCodeCircuit) -> np.ndarray:
    """
    Map each IBM ancilla index to its stim (x, y) coordinate (halved integer).

    IBM chip positions use (row, col) with qubit_number = row*10 + col.  The
    IBM surface code sits as a diamond on that grid; rotating by 45° and
    translating so the top-left ancilla sits at (0, 0) gives stim's
    rotated-memory-z frame (halved):

        stim_x = ((r - c) - u_min) // 2
        stim_y = ((r + c) - v_min) // 2

    where u_min / v_min are the minima over all IBM ancilla chip positions.
    """
    chip_rc = [(p // 10, p % 10) for p in sc.ancilla_physical]
    us = np.array([r - c for r, c in chip_rc], dtype=np.int64)
    vs = np.array([r + c for r, c in chip_rc], dtype=np.int64)
    xy = np.stack([(us - us.min()) // 2, (vs - vs.min()) // 2], axis=1)
    return xy.astype(np.int64)


@lru_cache(maxsize=None)
def _cached_alignment(distance: int, rounds: int, noise: float) -> StimAlignment:
    # Build a minimal SurfaceCodeCircuit just for its layout tables (T=0 skips
    # the qiskit circuit construction). Callers that already have one can use
    # `build_stim_alignment` directly.
    sc = SurfaceCodeCircuit(distance=distance, T=0)
    return _build(sc, rounds, noise)


def build_stim_alignment(
    sc: SurfaceCodeCircuit,
    rounds: int,
    noise: float = 1e-3,
) -> StimAlignment:
    """Build a StimAlignment for the given surface code circuit."""
    return _build(sc, rounds, noise)


def _build(sc: SurfaceCodeCircuit, rounds: int, noise: float) -> StimAlignment:
    num_anc = len(sc.ancilla_physical)

    ibm_xy = _ibm_ancilla_xy(sc)
    xy_to_ibm = {(int(x), int(y)): i for i, (x, y) in enumerate(ibm_xy)}

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=sc.distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
    )
    coords = circuit.get_detector_coordinates()

    # Group detectors by their time slice, preserving stim's internal order
    # (which is just the detector id within each slice).
    dets_by_t: dict[int, list[tuple[int, int, int]]] = {}
    for det_id in range(circuit.num_detectors):
        x, y, t = coords[det_id]
        dets_by_t.setdefault(int(t), []).append(
            (det_id, int(x) // 2, int(y) // 2)
        )

    ts = sorted(dets_by_t)
    if len(ts) < 2:
        raise ValueError(f"Expected at least 2 detector time slices, got {len(ts)}")

    # First slice: Z-only detectors. (For rotated_memory_z the last slice is
    # also Z-only and has the same layout as the first.)
    def _resolve(slice_entries):
        out = []
        for _, x, y in slice_entries:
            if (x, y) not in xy_to_ibm:
                raise KeyError(
                    f"Stim detector at halved xy ({x}, {y}) has no matching "
                    f"IBM ancilla — IBM layout and stim layout disagree."
                )
            out.append(xy_to_ibm[(x, y)])
        return np.array(out, dtype=np.int64)

    ibm_z_order = _resolve(dets_by_t[ts[0]])

    # Middle slice: all ancillas. Only meaningful when rounds >= 2.
    if rounds >= 2 and len(ts) >= 3:
        ibm_middle_order = _resolve(dets_by_t[ts[1]])
        if len(ibm_middle_order) != num_anc:
            raise ValueError(
                f"Middle round has {len(ibm_middle_order)} detectors, expected {num_anc}"
            )
    else:
        # No middle round; leave an identity permutation as a placeholder.
        ibm_middle_order = np.arange(num_anc, dtype=np.int64)

    # detector_is_z: 1.0 for Z-type, 0.0 for X-type, in stim detector order.
    n_z = len(ibm_z_order)
    det_is_z = [1.0] * n_z
    for _ in range(1, rounds):
        for anc_i in ibm_middle_order:
            det_is_z.append(0.0 if anc_i in sc.x_type else 1.0)
    det_is_z.extend([1.0] * n_z)
    detector_is_z = np.array(det_is_z, dtype=np.float32)

    return StimAlignment(
        circuit=circuit,
        rounds=rounds,
        ibm_middle_order=ibm_middle_order,
        ibm_z_order=ibm_z_order,
        ibm_ancilla_xy=ibm_xy.astype(np.float32),
        detector_is_z=detector_is_z,
    )


def ibm_detections_to_stim_order(
    detections_3d: np.ndarray,
    ibm_middle_order: np.ndarray,
    ibm_z_order: np.ndarray,
) -> np.ndarray:
    """
    Reorder IBM-indexed detections (shape `(shots, rounds+1, num_anc)`) into
    stim's detector ordering (flat, shape `(shots, num_detectors)`).
    """
    parts = [detections_3d[:, 0, ibm_z_order]]
    for r in range(1, detections_3d.shape[1] - 1):
        parts.append(detections_3d[:, r, ibm_middle_order])
    parts.append(detections_3d[:, -1, ibm_z_order])

    return np.hstack(parts).astype(bool)
