import json
from collections import Counter
import numpy as np
import torch
from qiskit_ibm_runtime import RuntimeDecoder
from torch_geometric.nn.pool import knn_graph
from surface_code_miami import SurfaceCodeCircuit
from stim_alignment import build_stim_alignment
from data import get_sliding_window


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


class IBMJobDecoder:
    """
    Loads IBM hardware job results and produces GNN-ready batches
    in the same format as data.py Dataset.generate_batch().
    """

    def __init__(self, sc: SurfaceCodeCircuit, job_path: str,
                 simulator: bool = False, k: int = 20, dt: int = 2,
                 norm: float = torch.inf, batch_size: int = 256,
                 device: torch.device = None, sliding: bool = True):
        self.distance = sc.distance
        self.t = sc.T
        self.num_ancilla = len(sc.ancilla_physical)
        self.x_type = sc.x_type
        self.job_path = job_path
        self.simulator = simulator
        self.k = k
        self.dt = dt
        self.norm = norm
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.sliding = sliding
        
        # Stabilizer-to-data-qubit map for final syndrome reconstruction
        self._stabilizer_data = sc.stabilizer_data

        # Use Stim-aligned ancilla coordinates so IBM and DEM share the same frame.
        alignment = build_stim_alignment(sc, rounds=self.t)
        self._detector_coords = np.zeros((self.num_ancilla, 4), dtype=np.float32)
        for i in range(self.num_ancilla):
            logical_x, logical_y = alignment.ibm_ancilla_xy[i]
            is_z = 0.0 if i in sc.x_type else 1.0
            self._detector_coords[i] = [logical_x, logical_y, is_z, 1.0 - is_z]

    def _load_job_data(self):
        """
        Parse IBM job JSON into detection events and logical flips.
        """
        if hasattr(self, 'detections'):
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

        # ---- Detection events = XOR between consecutive syndrome rounds ---- 
        initial_det = initial ^ syndromes[:, :1, :]
        middle_det = syndromes[:, :-1, :] ^ syndromes[:, 1:, :]
        final_det = final_syndrome[:, np.newaxis, :] ^ syndromes[:, -1:, :]

        self.detections = np.concatenate([initial_det, middle_det, final_det], axis=1).astype(bool)

        # ---- Logical observable (top row) ----
        logical_qubits = list(range(self.distance))
        self.logical_flips = np.zeros(actual_shots, dtype=np.int32)
        for q in logical_qubits:
            self.logical_flips ^= final_state[:, q].astype(np.int32)

    def _get_node_features(self, detection_batch):
        """
        Convert detection events to node features [x, y, t, is_X, is_Z].
        Only fired detectors become nodes.

        Returns:
            coords: (n_nodes, 5) with columns [x, y, t_raw, is_X, is_Z].
                    t_raw is the raw round index; caller applies dt chunking.
            shot_idx: (n_nodes,) shot/row each node came from.
        """
        shot_idx, t_idx, anc_idx = np.where(detection_batch)
        coords = np.empty((shot_idx.size, 5), dtype=np.float32)
        coords[:, :2] = self._detector_coords[anc_idx, :2]
        coords[:, 2] = t_idx
        coords[:, 3:] = self._detector_coords[anc_idx, 2:]
        return coords, shot_idx.astype(np.int64)

    def get_edges(self, node_features, labels):
        """
        Returns edges between nodes. The edges are of shape [n_edges, 2].
        Use ord=torch.inf for the supremum norm, ord=2 for euclidean norm.
        """
        edge_index = knn_graph(node_features, self.k, batch=labels)
        delta = node_features[edge_index[1]] - node_features[edge_index[0]]
        edge_attr = torch.linalg.norm(delta, ord=self.norm, dim=1)
        edge_attr = 1 / edge_attr ** 2
        return edge_index, edge_attr

    def _sliding_window(self, node_features):
        return get_sliding_window(node_features, self.dt, self.t, time_col=2)

    def _ensure_event_filter(self):
        """Cache the has-event filtered arrays so we do it once, not per batch."""
        if hasattr(self, '_det_filtered'):
            return
        has_event = np.any(self.detections.reshape(len(self.detections), -1), axis=1)
        self._det_filtered = self.detections[has_event]
        self._flips_filtered = self.logical_flips[has_event]

    def generate_batch(self, indices=None):
        """
        Generates a batch of graphs. If indices is None, samples randomly from
        event-filtered shots; otherwise builds the batch from those indices.
        """
        self._load_job_data()
        self._ensure_event_filter()

        if indices is None:
            n = min(self.batch_size, len(self._det_filtered))
            indices = np.random.choice(len(self._det_filtered), n, replace=False)
        det_batch = self._det_filtered[indices]
        flips_batch = self._flips_filtered[indices]
        n = len(indices)

        # Build node features (vectorized across all shots in the batch)
        coords, batch_labels = self._get_node_features(det_batch)

        # Expand each detector event into every dt-wide window containing it.
        if self.sliding:
            split_pts = np.searchsorted(batch_labels, np.arange(1, n))
            node_features = np.split(coords, split_pts)
            node_features, chunk_labels = self._sliding_window(node_features)
            batch_labels = np.repeat(
                np.arange(n),
                [len(features) for features in node_features],
            )
            coords = np.vstack(node_features).astype(np.float32)
        else:
            chunk_labels = (coords[:, 2] // self.dt).astype(np.int64)
            coords[:, 2] = coords[:, 2] % self.dt

        # Map [batch, chunk] -> label integer
        label_map = np.column_stack([batch_labels, chunk_labels])
        label_map, counts = np.unique(label_map, axis=0, return_counts=True)
        labels = np.repeat(np.arange(counts.shape[0]), counts).astype(np.int64)

        # Move to GPU before knn_graph so the graph build runs on device.
        node_features = torch.from_numpy(coords).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)
        label_map = torch.from_numpy(label_map).to(self.device)
        flips = torch.from_numpy(flips_batch[:n, np.newaxis]).to(self.device)

        edge_index, edge_attr = self.get_edges(node_features, labels)

        return node_features, edge_index, labels, label_map, edge_attr, flips


def split_ibm_job(sc: SurfaceCodeCircuit, job_path: str, ratios, seed: int,
                  dt: int, k: int, batch_size: int, device=None,
                  simulator: bool = False, sliding: bool = True):
    """
    Parse an IBM job once and return one IBMJobDecoder per split ratio, each
    pre-loaded with its slice of detections and logical flips.
    """
    template = IBMJobDecoder(
        sc, job_path=job_path, simulator=simulator, dt=dt, k=k,
        batch_size=batch_size, device=device, sliding=sliding,
    )
    template._load_job_data()
    n_total = len(template.logical_flips)
    perm = np.random.RandomState(seed).permutation(n_total)

    splits = []
    offset = 0
    for r in ratios:
        n = int(n_total * r)
        idx = perm[offset:offset + n]
        # Skip __init__ (and its stim alignment / stabilizer setup), reuse
        # the template's already-computed state and only override the slices.
        ds = IBMJobDecoder.__new__(IBMJobDecoder)
        ds.__dict__.update(template.__dict__)
        ds.detections = template.detections[idx]
        ds.logical_flips = template.logical_flips[idx]
        # Drop the template's cached filter so each child rebuilds its own.
        ds.__dict__.pop('_det_filtered', None)
        ds.__dict__.pop('_flips_filtered', None)
        splits.append(ds)
        offset += n
    return splits


def concat_ibm_decoders(datasets):
    """
    Merge pre-loaded IBMJobDecoders into one, skipping __init__.
    Shares the template's (_detector_coords, etc.) state; overrides only
    detections and logical_flips with the concatenated arrays. The
    event-filter cache is cleared so the merged dataset rebuilds its own.
    """
    if len(datasets) == 1:
        return datasets[0]
    template = datasets[0]
    ds = IBMJobDecoder.__new__(IBMJobDecoder)
    ds.__dict__.update(template.__dict__)
    ds.detections = np.concatenate([d.detections for d in datasets], axis=0)
    ds.logical_flips = np.concatenate([d.logical_flips for d in datasets], axis=0)
    ds.__dict__.pop('_det_filtered', None)
    ds.__dict__.pop('_flips_filtered', None)
    return ds


def prepare_real_datasets(sc: SurfaceCodeCircuit, train_jobs: list[str], *,
                          dt: int, k: int, batch_size: int, device=None,
                          test_ratio: float = 0.10, val_ratio: float = 0.10,
                          seed: int = 42, verbose: bool = True,
                          sliding: bool = True):
    """Build (real_train, real_val, real_test) from one or more IBM job files.

    Each job is split into train/val/test with the same ratios, then the
    matching splits are concatenated across jobs.
    """
    train_ratio = 1.0 - test_ratio - val_ratio
    train_parts = []
    val_parts = []
    test_parts = []
    per_job_sizes = []

    for i, job in enumerate(train_jobs):
        t_i, v_i, test_i = split_ibm_job(
            sc, job, ratios=[train_ratio, val_ratio, test_ratio], seed=seed + i,
            dt=dt, k=k, batch_size=batch_size, device=device, sliding=sliding,
        )
        train_parts.append(t_i)
        val_parts.append(v_i)
        test_parts.append(test_i)
        per_job_sizes.append((job, len(t_i.logical_flips),
                              len(v_i.logical_flips), len(test_i.logical_flips)))

    real_train = concat_ibm_decoders(train_parts)
    real_val = concat_ibm_decoders(val_parts)
    real_test = concat_ibm_decoders(test_parts)

    if verbose:
        for job, nt, nv, ntst in per_job_sizes:
            print(f"  {job}  -> train={nt}, val={nv}"
                  + (f", test={ntst}" if ntst else ""))
        print(f"Real shots - train: {len(real_train.logical_flips)}, "
              f"val: {len(real_val.logical_flips)}, "
              f"test: {len(real_test.logical_flips)}")

    return real_train, real_val, real_test


def evaluate_dataset(model, dataset, n_batches: int = 20,
                     all_shots: bool = False, threshold: float = 0.5) -> dict:
    """
    Per-class + overall accuracy.

    If all_shots=True, deterministically iterate every shot exactly once
    (event shots through the model, no-detection shots tallied as pred=0).
    Otherwise sample n_batches random batches.
    """
    model.eval()
    n0, n1, c0, c1 = 0, 0, 0, 0

    def _tally(flips, pred):
        nonlocal n0, n1, c0, c1
        is0 = flips == 0
        is1 = flips == 1
        n0 += is0.sum().item()
        n1 += is1.sum().item()
        c0 += ((pred == flips) & is0).sum().item()
        c1 += ((pred == flips) & is1).sum().item()

    with torch.no_grad():
        if all_shots:
            dataset._load_job_data()
            dataset._ensure_event_filter()
            n_event = len(dataset._det_filtered)
            for start in range(0, n_event, dataset.batch_size):
                idx = np.arange(start, min(start + dataset.batch_size, n_event))
                x, ei, lab, lm, ea, flips = dataset.generate_batch(indices=idx)
                out = model.forward(x, ei, ea, lab, lm)
                _tally(flips, (out >= threshold).to(flips.dtype))

            n_no_event = len(dataset.logical_flips) - len(dataset._flips_filtered)
            n1_no_event = int(dataset.logical_flips.sum() - dataset._flips_filtered.sum())
            n0_no_event = n_no_event - n1_no_event
            n0 += n0_no_event
            n1 += n1_no_event
            c0 += n0_no_event
        else:
            for _ in range(n_batches):
                x, ei, lab, lm, ea, flips = dataset.generate_batch()
                out = model.forward(x, ei, ea, lab, lm)
                _tally(flips, (out >= threshold).to(flips.dtype))

    total = n0 + n1
    return {
        "acc": (c0 + c1) / total,
        "acc_0": c0 / n0,
        "acc_1": c1 / n1,
        "n_0": n0,
        "n_1": n1,
    }


def tune_threshold(model, dataset, thresholds=None, all_shots: bool = True,
                   n_batches: int = 20) -> tuple[float, dict]:
    """
    Choose the threshold with best accuracy on a validation dataset.
    If all_shots=True, no-detection shots are included with model score 0.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        if all_shots:
            dataset._load_job_data()
            dataset._ensure_event_filter()
            n_event = len(dataset._det_filtered)
            for start in range(0, n_event, dataset.batch_size):
                idx = np.arange(start, min(start + dataset.batch_size, n_event))
                x, ei, lab, lm, ea, flips = dataset.generate_batch(indices=idx)
                probs.append(model.forward(x, ei, ea, lab, lm).detach().cpu())
                labels.append(flips.detach().cpu())

            has_event = np.any(dataset.detections.reshape(len(dataset.detections), -1), axis=1)
            no_event_flips = dataset.logical_flips[~has_event, np.newaxis]
            if len(no_event_flips) > 0:
                probs.append(torch.zeros((len(no_event_flips), 1)))
                labels.append(torch.from_numpy(no_event_flips))
        else:
            for _ in range(n_batches):
                x, ei, lab, lm, ea, flips = dataset.generate_batch()
                probs.append(model.forward(x, ei, ea, lab, lm).detach().cpu())
                labels.append(flips.detach().cpu())

    probs = torch.vstack(probs)
    labels = torch.vstack(labels)
    best_threshold = 0.5
    best_metrics = None

    for threshold in thresholds:
        pred = (probs >= float(threshold)).to(labels.dtype)
        is0 = labels == 0
        is1 = labels == 1
        n0 = is0.sum().item()
        n1 = is1.sum().item()
        c0 = ((pred == labels) & is0).sum().item()
        c1 = ((pred == labels) & is1).sum().item()
        total = n0 + n1
        metrics = {
            "acc": (c0 + c1) / total,
            "acc_0": c0 / n0 if n0 else 0.0,
            "acc_1": c1 / n1 if n1 else 0.0,
            "n_0": n0,
            "n_1": n1,
        }
        if best_metrics is None or metrics["acc"] > best_metrics["acc"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics

if __name__ == "__main__":

    D, T = 3, 10
    JOB = "jobs/dist3/job_d3_T10_shots100000_d7b87q15a5qc73dn58rg_.json"

    sc = SurfaceCodeCircuit(distance=D, T=T)
    dataset = IBMJobDecoder(sc, job_path=JOB, dt=2, k=20)
    dataset._load_job_data()

    print(f"Raw logical flip rate: {dataset.logical_flips.mean():.3f}")
    n_correct_trivial = (dataset.logical_flips == 0).sum()
    print(f"Trivial decoder accuracy: {n_correct_trivial / len(dataset.logical_flips):.3f}")

    det_flat = dataset.detections.reshape(len(dataset.detections), -1)
    print(f"Shots with any detection: {np.any(det_flat, axis=1).mean():.3f}")
    print(f"Mean detectors fired: {det_flat.sum(axis=1).mean():.1f}")
    np.set_printoptions(linewidth=200)

    # Per-round detection rates: each value is the detection rate for one ancilla (averaged over all shots)
    for r in range(dataset.detections.shape[1]):
        print(f"Round {r}: {dataset.detections[:, r, :].mean(axis=0).round(2)}")
