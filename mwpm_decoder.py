"""
mwpm_decoder.py
---------------
MWPM (Minimum Weight Perfect Matching) decoder for the repetition code on IBM hardware.

Decoding pipeline
-----------------
1. Load IBM Sampler job data (or Aer simulator counts) and convert bitstrings to
   detection events on a (t+1) × (d-1) spacetime detector grid.
2. Subsample sub-codes by sliding a spatial window of width (d-1) over the d=17
   ancilla array.
3. Build a PyMatching graph whose edge weights are derived from pairwise detection
   correlations estimated directly from the data.
4. Run batch MWPM decoding and compute logical error rates per sub-code.

Boundary edge weights
---------------------
For each boundary node i, all non-boundary incident edges (space-like, time-like,
diagonal) carry probabilities p_ij estimated from correlations.  Their combined
effect on the marginal detection rate <x_i> is given by the XOR-channel fold:

    p_{i,Σ} = g(p_{ij_k}, ..., g(p_{ij_2}, p_{ij_1}))
    g(p, q)  = p + q − 2pq  (probability an odd number of the events occur)

The missing boundary-edge probability that accounts for the rest of <x_i> is then:

    p_iB = (<x_i> − p_{i,Σ}) / (1 − 2 p_{i,Σ})

This is Eqn. (12) / (13) from the Spitz et al. detector-error-model paper and
gives a principled, data-driven estimate of the single-qubit error rate at each
code boundary.
"""

import numpy as np
import pymatching
from ibm_utils import parse_ibm_job


class MWPMDecoder:
    """
    MWPM decoder for repetition-code syndrome data from IBM hardware or Aer simulator.

    The matching graph has (t+1) × (d-1) detector nodes arranged in a 2-D
    spacetime grid.  Rows correspond to time slices (0 = first syndrome round,
    t = final readout syndrome); columns correspond to ancilla qubits.

    Three classes of bulk edges are added:
      - Space-like:  adjacent ancillas within the same time slice.
      - Time-like:   same ancilla across adjacent time slices.
      - Diagonal:    adjacent-time, adjacent-space (captures hook errors from CNOTs).

    Boundary half-edges connect the leftmost and rightmost node of each row to
    the virtual boundary node.  The left boundary carries fault_ids={0}, so a
    matching that crosses it is recorded in predictions[:, 0] and counted as a
    logical flip prediction.
    """

    def __init__(
        self,
        distance: int,
        t: int,
        job_path: str,
        simulator: bool,
        shots: int,
        ancilla_physical: list[int] = None,
        data_physical: list[int] = None,
        x_type: set[int] = None,
        code_type: str = "repetition",
    ) -> None:
        """
        Parameters
        ----------
        distance : int
            Physical code distance (number of data qubits for repetition, or nominal distance for surface).
        t : int
            Number of syndrome measurement rounds.
        job_path : str
            Path to the JSON file containing job results (RuntimeEncoder format).
        simulator : bool
            True if data comes from an Aer simulator (uses get_counts() API).
        shots : int
            Nominal shot count; actual count is inferred from the expanded arrays.
        ancilla_physical : list[int], optional
            Physical positions of ancilla qubits (for surface code).
        data_physical : list[int], optional
            Physical positions of data qubits (for surface code).
        x_type : set[int], optional
            Indices of X-type ancillas (0-based in ancilla list, for surface code).
        code_type : str
            "repetition" or "surface" to choose the decoding mode.
        """
        self.distance = distance
        self.t = t
        self.job_path = job_path
        self.simulator = simulator
        self.shots = shots
        self.ancilla_physical = ancilla_physical
        self.data_physical = data_physical
        self.x_type = x_type
        self.code_type = code_type

        if code_type == "surface":
            if ancilla_physical is None or data_physical is None or x_type is None:
                raise ValueError("For surface code, ancilla_physical, data_physical, and x_type must be provided")
            self.n_measures = len(ancilla_physical)
            self.data_idx = {phys: i for i, phys in enumerate(data_physical)}
            self.data_set = set(data_physical)
            # Precompute data measured by each ancilla
            self.data_measured = []
            for anc_i, anc_p in enumerate(ancilla_physical):
                is_x = anc_i in x_type
                directions = [-1, -10, 10, 1] if is_x else [-1, 10, -10, 1]  # X_ORDER or Z_ORDER
                measured = []
                for d in directions:
                    neighbour = anc_p + d
                    if neighbour in self.data_set:
                        measured.append(self.data_idx[neighbour])
                self.data_measured.append(measured)
        else:
            self.n_measures = distance - 1

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_job_data(self) -> None:
        """
        Load, parse, and preprocess syndrome data from the job JSON file.

        IBM bitstrings are MSB-first (most recently measured bit at index 0),
        so we reverse along the bit axis to obtain qubit-0-first, round-0-first
        ordering.

        Because no mid-circuit reset is used, ancilla measurements accumulate
        across rounds.  The actual per-round syndrome is obtained by XOR-diffing
        consecutive raw measurements (equivalent to virtual resets).

        Detection events are changes in the (virtual-reset) syndrome between
        consecutive rounds, including:
          - A virtual all-zero initial syndrome (valid for logical-0 preparation).
          - t syndrome rounds.
          - A final syndrome inferred from the data-qubit readout.

        Populates
        ---------
        self.partitions : dict[(d, i)] → np.ndarray, shape (shots, (t+1)*(d-1))
            Flattened boolean detection-event vectors for each sub-code.
        self.logical_flips : dict[(d, i)] → np.ndarray, shape (shots,), dtype bool
            True when the left-boundary data qubit was measured as |1⟩.
        """
        if self.code_type == "surface":
            n_data = len(self.data_physical)
        else:
            n_data = self.distance
        final_state, syndromes = parse_ibm_job(
            self.job_path, self.t, n_data, self.n_measures, self.simulator
        )

        actual_shots = final_state.shape[0]

        # Virtual initial syndrome (all-zero; valid for |0⟩_L preparation).
        initial_syndrome = np.zeros((actual_shots, self.n_measures), dtype=np.uint8)

        if self.code_type == "surface":
            # Final syndrome: inferred from data readout for each ancilla
            final_syndrome = np.zeros((actual_shots, self.n_measures), dtype=np.uint8)
            for anc_i in range(self.n_measures):
                measured_data = self.data_measured[anc_i]
                if measured_data:
                    final_syndrome[:, anc_i] = np.sum(final_state[:, measured_data], axis=1) % 2
        else:
            # Final syndrome: XOR of adjacent data qubits after readout.
            final_syndrome = final_state[:, :-1] ^ final_state[:, 1:]

        # Stack into (shots, t+2, n_measures): initial | t rounds | final.
        syndrome_matrix = np.concatenate(
            [
                initial_syndrome,
                syndromes.reshape(actual_shots, -1),
                final_syndrome,
            ],
            axis=1,
        )
        reshaped = syndrome_matrix.reshape(actual_shots, self.t + 2, self.n_measures)

        # Detection events = XOR between consecutive syndrome rounds.
        # Shape: (shots, t+1, n_measures).
        detections = np.diff(reshaped, axis=1).astype(bool)

        # Subsample sub-codes or use full for surface
        self.partitions: dict = {}
        self.logical_flips: dict = {}

        if self.code_type == "surface":
            self.partitions[(self.distance, 0)] = detections.reshape(actual_shots, -1)
            # Logical observable: parity of data qubits (for logical X in X-basis)
            self.logical_flips[(self.distance, 0)] = np.sum(final_state, axis=1) % 2 == 1
        else:
            for d in range(3, self.distance + 1, 2):
                n_meas = d - 1
                for i in range(self.distance - d + 1):
                    part = detections[:, :, i : i + n_meas]
                    self.partitions[(d, i)] = part.reshape(actual_shots, -1)
                    self.logical_flips[(d, i)] = final_state[:, i] == 1

    # ------------------------------------------------------------------
    # Edge-weight computation
    # ------------------------------------------------------------------

    def _error_correlation_matrix(
        self, detections: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate pairwise error probabilities p_ij from detection-event data.

        Uses the formula from / Spitz et al. (2018):

            p_ij = 0.5 − 0.5 * sqrt(1 − 4*num / den)

        where
            num = <x_i x_j> − <x_i><x_j>
            den = 1 − 2<x_i> − 2<x_j> + 4<x_i x_j>

        Parameters
        ----------
        detections : np.ndarray, shape (shots, N)
            Binary detection-event matrix.

        Returns
        -------
        pij : np.ndarray, shape (N, N)
            Symmetric pairwise error probability matrix (diagonal = 0).
        mean_i : np.ndarray, shape (N,)
            Marginal detection probability <x_i> of each detector node.
        """
        x = detections.astype(np.float64)
        mean_i = x.mean(axis=0)               # <x_i>,  shape (N,)
        mean_ij = (x.T @ x) / x.shape[0]     # <x_i x_j>, shape (N, N)

        numerator = mean_ij - np.outer(mean_i, mean_i)
        denominator = (
            1 - 2 * mean_i[:, None] - 2 * mean_i[None, :] + 4 * mean_ij
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            sqrt_term = np.sqrt(1 - 4 * numerator / denominator)
            pij = 0.5 - 0.5 * sqrt_term

        pij = np.where(np.isfinite(pij), pij, 0.0)
        np.fill_diagonal(pij, 0.0)
        return pij, mean_i

    @staticmethod
    def _xor_fold(probs) -> float:
        """
        Fold a list of error probabilities under the XOR channel.

        Computes the probability that an odd number of independent binary events
        occur, using the recursive identity:

            g(p, q) = p + q − 2pq

        An empty list returns 0 (no contribution from an empty edge set).
        """
        result = 0.0
        for p in probs:
            result = result + p - 2.0 * result * p
        return result

    def _compute_boundary_prob(
        self,
        node: int,
        row_len: int,
        pij: np.ndarray,
        mean_i: np.ndarray,
    ) -> float:
        """
        Compute the boundary-edge error probability for a boundary node.

        Collects the pij values for all non-boundary edges incident on `node`
        (space-like, time-like, diagonal), XOR-folds them to get p_{i,Σ}, then
        inverts via:

            p_iB = (<x_i> − p_{i,Σ}) / (1 − 2 p_{i,Σ})

        This is Eqn. (13) of the Spitz et al. detector-error-model paper.

        Parameters
        ----------
        node : int
            Detector node index (= t_index * row_len + offset).
        row_len : int
            Number of ancillas per time slice (= d − 1).
        pij : np.ndarray, shape (N, N)
            Pairwise error probability matrix from _error_correlation_matrix.
        mean_i : np.ndarray, shape (N,)
            Marginal detection probabilities.

        Returns
        -------
        p_boundary : float
            Estimated boundary error probability, clipped to [1e-7, 1 − 1e-7].
        """
        t_index = node // row_len
        offset = node % row_len
        incident = []

        # --- Space-like neighbors (same time slice) ---
        if offset > 0:
            incident.append(float(pij[node, node - 1]))
        if offset < row_len - 1:
            incident.append(float(pij[node, node + 1]))

        # --- Time-like neighbors (same spatial offset, adjacent time slices) ---
        # Downward (t_index → t_index+1): exists for t_index < self.t
        if t_index < self.t:
            incident.append(float(pij[node, node + row_len]))
        # Upward (t_index → t_index-1): exists for t_index > 0
        if t_index > 0:
            incident.append(float(pij[node, node - row_len]))

        # --- Diagonal neighbors (adjacent time slices, offset ± 1) ---
        # Diagonal edges exist for t_index in 0..t-1 (downward only in the loop).
        #
        # Forward-down: edge from (t_index, offset) to (t_index+1, offset+1)
        if t_index < self.t and offset + 1 < row_len:
            incident.append(float(pij[node, node + row_len + 1]))
        # Backward-down: edge from (t_index, offset) to (t_index+1, offset-1)
        if t_index < self.t and offset - 1 >= 0:
            incident.append(float(pij[node, node + row_len - 1]))
        # Forward-up: node is destination of backward-down from (t_index-1, offset+1)
        #   i.e., edge (t_index-1, offset+1) → (t_index, offset)
        if t_index > 0 and offset + 1 < row_len:
            incident.append(float(pij[node, node - row_len + 1]))
        # Backward-up: node is destination of forward-down from (t_index-1, offset-1)
        #   i.e., edge (t_index-1, offset-1) → (t_index, offset)
        if t_index > 0 and offset - 1 >= 0:
            incident.append(float(pij[node, node - row_len - 1]))

        p_sigma = self._xor_fold(incident)

        denom = 1.0 - 2.0 * p_sigma
        if abs(denom) < 1e-10:
            # Degenerate case: return a small but finite probability.
            return 1e-7

        p_boundary = (float(mean_i[node]) - p_sigma) / denom
        return float(np.clip(p_boundary, 1e-7, 1.0 - 1e-7))

    # ------------------------------------------------------------------
    # Matching graph construction
    # ------------------------------------------------------------------

    def get_edges(self, detections: np.ndarray, d: int) -> pymatching.Matching:
        """
        Build the PyMatching graph for a sub-code of distance d.

        For surface code, uses correlation-based edges between ancillas that share data qubits.
        For repetition, uses 1D adjacency.

        Parameters
        ----------
        detections : np.ndarray, shape (shots, (t+1)*n_measures)
            Flattened detection-event vectors.
        d : int
            Sub-code distance.

        Returns
        -------
        matcher : pymatching.Matching
        """
        row_len = self.n_measures
        pij, mean_i = self._error_correlation_matrix(detections)

        # Clip to (0, 1) before taking log; pij values are in [0, 0.5].
        pij_safe = np.where(pij > 0, pij, 1e-7)
        weights = -np.log(pij_safe)

        matcher = pymatching.Matching()

        if self.code_type == "surface":
            # Space-like edges: between ancillas that share data qubits
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    shared = set(self.data_measured[i]) & set(self.data_measured[j])
                    if shared:
                        matcher.add_edge(
                            i,
                            j,
                            weight=weights[i, j],
                            fault_ids=shared,
                            merge_strategy="replace",
                        )

            # Time-like edges
            for t_index in range(self.t):
                for offset in range(row_len):
                    i = t_index * row_len + offset
                    j = i + row_len
                    matcher.add_edge(
                        i,
                        j,
                        weight=weights[i, j],
                        merge_strategy="replace",
                    )

            # Diagonal edges (adjacent space and time)
            for t_index in range(self.t):
                for offset in range(row_len):
                    i = t_index * row_len + offset
                    # Forward diagonal
                    if offset + 1 < row_len:
                        matcher.add_edge(
                            i,
                            i + row_len + 1,
                            weight=weights[i, i + row_len + 1],
                            merge_strategy="replace",
                        )
                    # Backward diagonal
                    if offset - 1 >= 0:
                        matcher.add_edge(
                            i,
                            i + row_len - 1,
                            weight=weights[i, i + row_len - 1],
                            merge_strategy="replace",
                        )

            # Boundary half-edges for surface code
            rows = [p // 10 for p in self.ancilla_physical]
            min_row = min(rows)
            max_row = max(rows)
            for t_index in range(self.t + 1):
                for offset in range(row_len):
                    node = t_index * row_len + offset
                    anc_p = self.ancilla_physical[offset]
                    row = anc_p // 10
                    if row == min_row or row == max_row:  # Top/bottom boundaries for logical X
                        p_boundary = self._compute_boundary_prob(node, row_len, pij, mean_i)
                        matcher.add_boundary_edge(
                            node,
                            weight=-np.log(p_boundary),
                            fault_ids={0},
                            merge_strategy="replace",
                        )
        else:
            # Original repetition code logic
            row_len = d - 1
            # Space-like edges
            for t_index in range(self.t + 1):
                row_start = t_index * row_len
                for j in range(row_len - 1):
                    i = row_start + j
                    matcher.add_edge(
                        i,
                        i + 1,
                        weight=weights[i, i + 1],
                        fault_ids={j + 1},
                        merge_strategy="replace",
                    )

            # Time-like edges
            for t_index in range(self.t):
                for offset in range(row_len):
                    i = t_index * row_len + offset
                    j = i + row_len
                    matcher.add_edge(
                        i,
                        j,
                        weight=weights[i, j],
                        merge_strategy="replace",
                    )

            # Diagonal edges
            for t_index in range(self.t):
                row_start = t_index * row_len
                for offset in range(row_len):
                    i = row_start + offset
                    if offset + 1 < row_len:
                        matcher.add_edge(
                            i,
                            i + row_len + 1,
                            weight=weights[i, i + row_len + 1],
                            merge_strategy="replace",
                        )
                    if offset - 1 >= 0:
                        matcher.add_edge(
                            i,
                            i + row_len - 1,
                            weight=weights[i, i + row_len - 1],
                            merge_strategy="replace",
                        )

            # Boundary half-edges
            for t_index in range(self.t + 1):
                row_start = t_index * row_len
                left_node = row_start
                right_node = row_start + row_len - 1

                p_left = self._compute_boundary_prob(left_node, row_len, pij, mean_i)
                p_right = self._compute_boundary_prob(right_node, row_len, pij, mean_i)

                matcher.add_boundary_edge(
                    left_node,
                    weight=-np.log(p_left),
                    fault_ids={0},
                    merge_strategy="replace",
                )
                matcher.add_boundary_edge(
                    right_node,
                    weight=-np.log(p_right),
                    fault_ids={row_len},
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

        Shots with no detection events (trivial syndromes) are counted as
        correctly decoded without running MWPM (the all-zero correction is
        trivially right for logical-0 preparation).

        predictions[:, 0] counts how many times the matching crosses the left
        boundary (fault_id=0), which is the logical observable for this sub-code.

        Parameters
        ----------
        matcher : pymatching.Matching
        detections : np.ndarray, shape (shots, N), dtype bool
        logical_flips : np.ndarray, shape (shots,), dtype bool

        Returns
        -------
        logical_accuracy : float
        logical_accuracy_err : float
            Binomial standard error sqrt(p(1-p)/n).
        """
        nontrivial = np.any(detections, axis=1)
        detections_nt = detections[nontrivial]
        flips_nt = logical_flips[nontrivial]

        predictions = matcher.decode_batch(detections_nt)
        predicted = predictions[:, 0]  # left-boundary crossing count

        correct = np.sum(flips_nt == predicted)
        trivial_count = np.sum(~nontrivial)
        total = detections.shape[0]

        logical_accuracy = (correct + trivial_count) / total
        logical_accuracy_err = np.sqrt(
            logical_accuracy * (1 - logical_accuracy) / total
        )
        return logical_accuracy, logical_accuracy_err

    def decode(self) -> dict:
        """
        Run the full decoding pipeline and return logical error rates.

        Loads data, constructs a matching graph per sub-code, runs MWPM decoding,
        and collects results.

        Returns
        -------
        P_L : dict[(d, i)] → (p_l, err, pdet_mean)
            p_l : float
                Logical error rate  P_L = 1 − accuracy.
            err : float
                Binomial standard error on the accuracy estimate.
            pdet_mean : float
                Mean detection probability for this (d, i) sub-code.
        """
        self._load_job_data()
        P_L = {}

        for d, i in self.partitions:
            detections = self.partitions[(d, i)]
            logical_flips = self.logical_flips[(d, i)]
            matcher = self.get_edges(detections, d)
            pdet_mean = float(detections.mean())
            accuracy, accuracy_err = self._evaluate_predictions(
                matcher, detections, logical_flips
            )
            P_L[(d, i)] = (1 - accuracy, accuracy_err, pdet_mean)

            print(
                f"d={d}, i={i}, pdet={pdet_mean:.4f}, "
                f"P_L={1 - accuracy:.6f} ± {accuracy_err:.6f}"
            )

        return P_L
