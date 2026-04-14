"""
mwpm_decoder.py
---------------
MWPM (Minimum Weight Perfect Matching) decoder for the surface code on IBM hardware.

Decoding pipeline
-----------------
1. Load IBM job data and convert to detection events on a (t+1) × num_ancilla
   spacetime detector grid.
2. Extract Z-type detection events.
3. Build a PyMatching graph whose edge weights are derived from pairwise
   detection correlations (Spitz et al.).
4. Run batch MWPM decoding and compute logical error rates.

Detection event boundaries
--------------------------
- Z-type: initial = 0 (known eigenstate |0⟩), final = parity of data qubits in support.
- X-type: initial = syndromes[:, 0, :] (absorbs random eigenvalue), final = last round.
  (X-type detections are computed but not currently decoded.)

Matching graph topology
-----------------------
The Z-type spacetime grid has (t+1) rows × n_z_anc columns.
Node index: node = t_index * n_z_anc + anc_offset.

Three classes of bulk edges:
  - Space-like:  ancillas that share a data qubit, within the same time slice.
  - Time-like:   same ancilla across adjacent time slices.
  - Diagonal:    adjacent in both space and time (hook errors).

Boundary half-edges connect ancillas with unshared data qubits to the virtual
boundary. Ancillas whose unshared qubits overlap with the Z logical operator
(top row) are tagged with fault_ids={0}.
"""

import json
from collections import Counter

import numpy as np
import pymatching
from qiskit_ibm_runtime import RuntimeDecoder

from surface_code_miami import SurfaceCodeCircuit, X_ORDER, Z_ORDER


def parse_ibm_job(job_path, t, n_data, n_measures, simulator=False):
    """
    Parse IBM job JSON into final data-qubit states and virtual-reset syndromes.

    Handles both simulator (get_counts) and hardware (per-register bitstrings),
    expands compressed counts, converts IBM's MSB-first ordering to qubit-0-first
    ordering, and applies no-reset XOR diffing on syndrome rounds.

    Returns
    -------
    final_state   : np.ndarray, shape (shots, n_data), dtype uint8
        Final Z-basis readout of data qubits (qubit-0 first).
    syndromes     : np.ndarray, shape (shots, t, n_measures), dtype uint8
        Virtual-reset syndromes where syndromes[:, 0, :] is the first raw
        measurement round and later rounds are XOR differences between
        consecutive raw rounds.
    """
    with open(job_path) as f:
        data = json.load(f, cls=RuntimeDecoder)

    if simulator:
        counts = data.get_counts()
    else:
        measure_regs = [f"round_{i}_measure_bit" for i in range(t)]
        all_regs = ["code_bit"] + measure_regs
        regs = {name: data[0].data[name].get_bitstrings() for name in all_regs}
        bitstrings = [
            "".join(bits) for bits in zip(*(regs[name] for name in all_regs))
        ]
        counts = Counter(bitstrings)

    final_state_list, syndromes_nr_list = [], []
    for bitstring, freq in counts.items():
        final_state_list.append((bitstring[:n_data], freq))
        syndromes_nr_list.append((bitstring[n_data:], freq))

    final_state = np.array([list(s[0]) for s in final_state_list], dtype=np.uint8)
    syndromes_nr = np.array([list(s[0]) for s in syndromes_nr_list], dtype=np.uint8)
    freqs = np.array([s[1] for s in final_state_list], dtype=int)

    final_state = np.repeat(final_state, freqs, axis=0)
    syndromes_nr = np.repeat(syndromes_nr, freqs, axis=0)

    # Reverse data-qubit order: IBM MSB-first -> qubit-0 first
    final_state = final_state[:, ::-1]

    # Reshape flat syndrome bits into rounds, then reverse ancilla bit order
    syndromes_nr = syndromes_nr.reshape(-1, t, n_measures)
    syndromes_nr = syndromes_nr[:, :, ::-1]

    # No-reset XOR diffing
    diff = (syndromes_nr[:, 1:, :] != syndromes_nr[:, :-1, :]).astype(np.uint8)
    first = syndromes_nr[:, :1, :]
    syndromes = np.concatenate([first, diff], axis=1)

    return final_state, syndromes


class MWPMDecoder:
    """
    MWPM decoder for surface-code syndrome data from IBM hardware or Aer simulator.

    The matching graph has (t+1) × n_anc detector nodes arranged in a 2-D
    spacetime grid. Rows correspond to time slices and columns correspond to
    Z-type ancillas.

    Three classes of bulk edges are added:
      - Space-like:  adjacent ancillas within the same time slice.
      - Time-like:   same ancilla across adjacent time slices.
      - Diagonal:    adjacent-time, adjacent-space (hook-error channels).

    Boundary half-edges connect boundary ancillas to the virtual boundary.
    One boundary class is tagged with fault_ids={0} and used as the logical
    observable for Z-basis memory decoding.
    """

    def __init__(
        self,
        sc: SurfaceCodeCircuit,
        job_path: str,
        simulator: bool = False,
        pij_threshold: float = 0.0,
    ) -> None:
        self.distance = sc.distance
        self.t = sc.T
        self.num_ancilla = len(sc.ancilla_physical)
        self.x_type = sc.x_type
        self.job_path = job_path
        self.simulator = simulator
        self.pij_threshold = pij_threshold

        # Indices of Z-type ancillas
        self.z_indices = sorted(i for i in range(self.num_ancilla) if i not in self.x_type)

        # Build stabilizer-to-data-qubit map
        self._stabilizer_data = {}
        for anc_i, anc_p in enumerate(sc.ancilla_physical):
            is_x = anc_i in sc.x_type
            order = X_ORDER if is_x else Z_ORDER
            neighbors = []
            for direction in order:
                nb = anc_p + direction
                if nb in sc.data_set:
                    neighbors.append(sc.data_idx[nb])
            self._stabilizer_data[anc_i] = neighbors

        # Build spatial adjacency: two Z ancillas are neighbors if they share
        # at least one data qubit.
        self._z_adj = self._build_adjacency(self.z_indices)

        # Data qubits shared between 2+ Z stabilizers (interior qubits).
        z_data_count: dict[int, int] = {}
        for anc_i in self.z_indices:
            for d_i in self._stabilizer_data[anc_i]:
                z_data_count[d_i] = z_data_count.get(d_i, 0) + 1
        shared_z_data = {d_i for d_i, c in z_data_count.items() if c >= 2}

        # Boundary = any Z ancilla with at least one unshared data qubit
        # (covers both rough-boundary weight-2 stabilizers AND smooth-boundary
        # weight-4 stabilizers that have data qubits on the code edge).
        z_logical_set = set(range(self.distance))  # top row = Z logical
        self._z_boundary = set()
        self._z_logical_boundary = set()
        for anc_i in self.z_indices:
            unshared = [d_i for d_i in self._stabilizer_data[anc_i]
                        if d_i not in shared_z_data]
            if unshared:
                self._z_boundary.add(anc_i)
                if any(d_i in z_logical_set for d_i in unshared):
                    self._z_logical_boundary.add(anc_i)

    def _build_adjacency(self, indices: list[int]) -> dict[int, list[int]]:
        """
        Build spatial adjacency list for a set of same-type ancillas.
        Two ancillas are adjacent if they share at least one data qubit.
        """
        data_to_anc: dict[int, list[int]] = {}
        for anc_i in indices:
            for d_i in self._stabilizer_data[anc_i]:
                data_to_anc.setdefault(d_i, []).append(anc_i)

        adj_sets: dict[int, set[int]] = {i: set() for i in indices}
        for anc_list in data_to_anc.values():
            for i, a in enumerate(anc_list):
                for b in anc_list[i + 1:]:
                    adj_sets[a].add(b)
                    adj_sets[b].add(a)

        return {i: sorted(adj_sets[i]) for i in indices}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_job_data(self) -> None:
        """
        Parse IBM job JSON into detection events and logical flips.

        IBM bitstrings are converted to qubit-ordered arrays by parse_ibm_job,
        including virtual-reset syndrome extraction via XOR differencing.

        Detection events are built from:
            - Initial reference syndrome.
            - Round-to-round syndrome changes.
            - Final reference syndrome inferred from data readout.

        Populates
        ---------
        self.detections : np.ndarray, shape (shots, t+1, num_ancilla), dtype bool
        self.logical_flips : np.ndarray, shape (shots,), dtype int32
        """
        if hasattr(self, "detections") and hasattr(self, "logical_flips"):
            return

        n_data = self.distance ** 2
        final_state, syndromes = parse_ibm_job(
            self.job_path,
            self.t,
            n_data,
            self.num_ancilla,
            self.simulator,
        )

        actual_shots = final_state.shape[0]

        # ---- Build initial and final syndrome references ----
        initial = np.zeros((actual_shots, 1, self.num_ancilla), dtype=np.uint8)
        final_syndrome = np.zeros((actual_shots, self.num_ancilla), dtype=np.uint8)

        for anc_i, data_indices in self._stabilizer_data.items():
            if anc_i in self.x_type:
                initial[:, 0, anc_i] = syndromes[:, 0, anc_i]
                final_syndrome[:, anc_i] = syndromes[:, -1, anc_i]
            else:
                parity = np.zeros(actual_shots, dtype=np.uint8)
                for d_i in data_indices:
                    parity ^= final_state[:, d_i]
                final_syndrome[:, anc_i] = parity

        # ---- Detection events ----
        initial_det = initial ^ syndromes[:, :1, :]
        middle_det = syndromes[:, :-1, :] ^ syndromes[:, 1:, :]
        final_det = final_syndrome[:, np.newaxis, :] ^ syndromes[:, -1:, :]

        self.detections = np.concatenate(
            [initial_det, middle_det, final_det], axis=1
        ).astype(bool)  # (shots, t+1, num_ancilla)

        # ---- Logical observable: parity of top-row data qubits ----
        logical_qubits = list(range(self.distance))
        self.logical_flips = np.zeros(actual_shots, dtype=np.int32)
        for q in logical_qubits:
            self.logical_flips ^= final_state[:, q].astype(np.int32)

    # ------------------------------------------------------------------
    # Edge-weight computation (Spitz et al.)
    # ------------------------------------------------------------------

    @staticmethod
    def _error_correlation_matrix(
        detections: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate pairwise error probabilities p_ij from detection-event data.

            p_ij = 0.5 - 0.5 * sqrt(1 - 4*num / den)

        where num = <x_i x_j> - <x_i><x_j>, den = 1 - 2<x_i> - 2<x_j> + 4<x_i x_j>.

        Returns (pij, mean_i).
        """
        x = detections.astype(np.float64)
        mean_i = x.mean(axis=0)
        mean_ij = (x.T @ x) / x.shape[0]

        numerator = mean_ij - np.outer(mean_i, mean_i)
        denominator = 1 - 2 * mean_i[:, None] - 2 * mean_i[None, :] + 4 * mean_ij

        with np.errstate(divide="ignore", invalid="ignore"):
            sqrt_term = np.sqrt(1 - 4 * numerator / denominator)
            pij = 0.5 - 0.5 * sqrt_term

        pij = np.where(np.isfinite(pij), pij, 0.0)
        np.fill_diagonal(pij, 0.0)
        return pij, mean_i

    @staticmethod
    def _xor_fold(probs) -> float:
        """Fold probabilities under XOR channel: g(p,q) = p + q - 2pq."""
        result = 0.0
        for p in probs:
            result = result + p - 2.0 * result * p
        return result

    def _compute_boundary_prob(
        self,
        node: int,
        n_anc: int,
        pij: np.ndarray,
        mean_i: np.ndarray,
        adj: dict[int, list[int]],
        anc_list: list[int],
    ) -> float:
        """
        Compute boundary-edge probability for a boundary node using XOR inversion.

            p_iB = (<x_i> - p_{i,Σ}) / (1 - 2*p_{i,Σ})

        Collects all non-boundary incident edges (space-like, time-like, diagonal).
        """
        t_index = node // n_anc
        offset = node % n_anc
        anc_i = anc_list[offset]
        anc_to_offset = {anc: i for i, anc in enumerate(anc_list)}
        incident = []

        # Space-like: same time slice, spatially adjacent
        for nb_anc in adj[anc_i]:
            nb_offset = anc_to_offset[nb_anc]
            nb_node = t_index * n_anc + nb_offset
            incident.append(float(pij[node, nb_node]))

        # Time-like: same ancilla, adjacent time
        if t_index < self.t:
            incident.append(float(pij[node, node + n_anc]))
        if t_index > 0:
            incident.append(float(pij[node, node - n_anc]))

        # Diagonal: adjacent time AND adjacent space
        for nb_anc in adj[anc_i]:
            nb_offset = anc_to_offset[nb_anc]
            if t_index < self.t:
                nb_node = (t_index + 1) * n_anc + nb_offset
                incident.append(float(pij[node, nb_node]))
            if t_index > 0:
                nb_node = (t_index - 1) * n_anc + nb_offset
                incident.append(float(pij[node, nb_node]))

        p_sigma = self._xor_fold(incident)
        denom = 1.0 - 2.0 * p_sigma
        if abs(denom) < 1e-10:
            return 1e-7

        p_boundary = (float(mean_i[node]) - p_sigma) / denom
        return float(np.clip(p_boundary, 1e-7, 1.0 - 1e-7))

    # ------------------------------------------------------------------
    # Matching graph construction
    # ------------------------------------------------------------------

    def get_edges(self, detections: np.ndarray, d: int) -> pymatching.Matching:
        """
        Build the PyMatching graph for Z-type detectors.

        Node layout
        -----------
        Nodes are indexed as node = t_index * n_anc + offset,
        where t_index in {0, ..., t} and offset indexes Z ancillas.

        Edges
        -----
        - Space-like  (t, i) -- (t, j): ancillas sharing a data qubit.
        - Time-like   (t, i) -- (t+1, i): same ancilla across rounds.
        - Diagonal    (t, i) -- (t+1, j): adjacent in time and space.
        - Boundary half-edges to virtual boundary.

        Parameters
        ----------
        detections : np.ndarray, shape (shots, (t+1)*n_anc), dtype bool
            Flattened Z-type detection events.
        d : int
            Code distance.
        """
        anc_list = self.z_indices
        adj = self._z_adj
        boundary_set = self._z_boundary
        logical_boundary_set = self._z_logical_boundary

        row_len = len(anc_list)
        pij, mean_i = self._error_correlation_matrix(detections)
        pij_safe = np.where(pij > self.pij_threshold, pij, 1e-7)
        weights = -np.log(pij_safe)
        anc_to_offset = {anc: offset for offset, anc in enumerate(anc_list)}

        matcher = pymatching.Matching()

        # --- Space-like edges ---
        for t_index in range(self.t + 1):
            for offset, anc_i in enumerate(anc_list):
                i = t_index * row_len + offset
                for nb_anc in adj[anc_i]:
                    nb_offset = anc_to_offset[nb_anc]
                    if nb_offset > offset:  # avoid duplicates
                        j = t_index * row_len + nb_offset
                        matcher.add_edge(
                            i, j,
                            weight=weights[i, j],
                            merge_strategy="replace",
                        )

        # --- Time-like edges ---
        for t_index in range(self.t):
            for offset in range(row_len):
                i = t_index * row_len + offset
                j = i + row_len
                matcher.add_edge(
                    i, j,
                    weight=weights[i, j],
                    merge_strategy="replace",
                )

        # --- Diagonal edges ---
        for t_index in range(self.t):
            for offset, anc_i in enumerate(anc_list):
                i = t_index * row_len + offset
                for nb_anc in adj[anc_i]:
                    nb_offset = anc_to_offset[nb_anc]
                    j = (t_index + 1) * row_len + nb_offset
                    matcher.add_edge(
                        i, j,
                        weight=weights[i, j],
                        merge_strategy="replace",
                    )

        # --- Boundary half-edges ---
        for t_index in range(self.t + 1):
            for offset, anc_i in enumerate(anc_list):
                if anc_i in boundary_set:
                    node = t_index * row_len + offset
                    p_b = self._compute_boundary_prob(
                        node, row_len, pij, mean_i, adj, anc_list,
                    )
                    if anc_i in logical_boundary_set:
                        matcher.add_boundary_edge(
                            node,
                            weight=-np.log(p_b),
                            fault_ids={0},
                            merge_strategy="replace",
                        )
                    else:
                        matcher.add_boundary_edge(
                            node,
                            weight=-np.log(p_b),
                            merge_strategy="replace",
                        )

        return matcher

    # ------------------------------------------------------------------
    # Decoding and evaluation
    # ------------------------------------------------------------------

    def _evaluate_predictions(
        self,
        matcher: pymatching.Matching,
        detections: np.ndarray,
        logical_flips: np.ndarray,
    ) -> tuple[float, float]:
        """
        Decode a batch of detection events and compute logical accuracy.

        Shots with no detection events are left as the all-zero prediction,
        matching the no-correction baseline behavior.

        Returns
        -------
        logical_accuracy : float
        logical_accuracy_err : float
            Binomial standard error sqrt(p(1-p)/n).
        """
        nontrivial = np.any(detections, axis=1)
        detections_nt = detections[nontrivial]
        flips_nt = logical_flips[nontrivial]

        if detections_nt.shape[0] > 0:
            predictions = matcher.decode_batch(detections_nt)
            predicted = predictions[:, 0]
            correct = np.sum(flips_nt == predicted)
        else:
            correct = 0

        trivial_count = np.sum(~nontrivial)
        total = detections.shape[0]

        logical_accuracy = (correct + trivial_count) / total
        logical_accuracy_err = np.sqrt(
            logical_accuracy * (1 - logical_accuracy) / total
        )
        return float(logical_accuracy), float(logical_accuracy_err)

    def decode(self) -> tuple[float, float]:
        """
        Run the full decoding pipeline.

        Uses Z-type detector events and compares predicted logical flips against
        top-row data-parity labels.

        Returns
        -------
        p_l : float
            Logical error rate.
        p_l_err : float
            Binomial standard error.
        """
        self._load_job_data()

        det_z = self.detections[:, :, self.z_indices].reshape(len(self.detections), -1)
        matcher = self.get_edges(det_z, self.distance)
        pdet_mean = float(det_z.mean())
        accuracy, accuracy_err = self._evaluate_predictions(
            matcher,
            det_z,
            self.logical_flips,
        )

        p_l = 1 - accuracy
        print(
            f"d={self.distance}, pdet={pdet_mean:.4f}, "
            f"Acc = {accuracy:.6f} ± {accuracy_err:.6f}, "
            f"P_L={p_l:.6f} ± {accuracy_err:.6f}, "
        )
        return p_l, accuracy_err


if __name__ == "__main__":

    D, T = 3, 10
    JOB = "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json"

    sc = SurfaceCodeCircuit(distance=D, T=T)
    decoder = MWPMDecoder(sc, job_path=JOB, pij_threshold=0.044)# pij_threshold=0.044 for d=3. pij_threshold=0 for d=5
    p_l, p_l_err = decoder.decode()