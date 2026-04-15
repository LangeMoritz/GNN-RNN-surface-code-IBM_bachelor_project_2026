import json
from collections import Counter
import numpy as np
import torch
from qiskit_ibm_runtime import RuntimeDecoder
from torch_geometric.nn.pool import knn_graph
from surface_code_miami import SurfaceCodeCircuit
from stim_alignment import build_stim_alignment


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
                 norm: float = torch.inf, batch_size: int = 2048,
                 device: torch.device = None):
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
        
        # Stabilizer-to-data-qubit map for final syndrome reconstruction
        # (now computed once inside SurfaceCodeCircuit).
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
        """
        node_features_list = []
        for shot_detections in detection_batch:
            fired = np.argwhere(shot_detections)
            if len(fired) == 0:
                continue
            coords = np.zeros((len(fired), 5), dtype=np.float32)
            for j, (t_idx, anc_idx) in enumerate(fired):
                coords[j, :2] = self._detector_coords[anc_idx, :2]
                coords[j, 2] = t_idx
                coords[j, 3:] = self._detector_coords[anc_idx, 2:]
            node_features_list.append(coords)
        return node_features_list

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

    def generate_batch(self):
        """
        Generates a batch of graphs.

        Returns:
            node_features: tensor of shape [n, 5] ([x, y, t, (stabilizer type)]).
            edge_index: tensor of shape [n_edges, 2].
            labels: tensor of shape [n].
            label_map: tensor of shape [n_graphs].
            edge_attr: tensor of shape [n_edges].
            flips: tensor of shape [batch size].
        """
        self._load_job_data()

        # Filter shots with at least one detection event
        has_event = np.any(self.detections.reshape(len(self.detections), -1), axis=1)
        det_filtered = self.detections[has_event]
        flips_filtered = self.logical_flips[has_event]

        n = min(self.batch_size, len(det_filtered))
        indices = np.random.choice(len(det_filtered), n, replace=False)
        det_batch = det_filtered[indices]
        flips_batch = flips_filtered[indices]

        # Build node features
        node_features_list = self._get_node_features(det_batch)

        # Chunk by dt
        all_nodes = []
        batch_labels = []
        chunk_labels = []
        for b_idx, coords in enumerate(node_features_list):
            chunks = coords[:, 2] // self.dt
            coords_copy = coords.copy()
            coords_copy[:, 2] = coords[:, 2] % self.dt
            all_nodes.append(coords_copy)
            batch_labels.extend([b_idx] * len(coords))
            chunk_labels.extend(chunks.astype(int).tolist())

        node_features = torch.from_numpy(np.vstack(all_nodes))
        batch_labels = np.array(batch_labels)
        chunk_labels = np.array(chunk_labels)

        # Map [batch, chunk] -> label integer
        label_map = np.column_stack([batch_labels, chunk_labels])
        label_map, counts = np.unique(label_map, axis=0, return_counts=True)
        labels = np.repeat(np.arange(counts.shape[0]), counts).astype(np.int64)
        label_map = torch.from_numpy(label_map)
        labels = torch.from_numpy(labels)

        edge_index, edge_attr = self.get_edges(node_features, labels)

        node_features = node_features.to(self.device)
        flips = torch.from_numpy(flips_batch[:n, np.newaxis]).to(self.device)
        labels = labels.to(self.device)
        label_map = label_map.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        return node_features, edge_index, labels, label_map, edge_attr, flips


def split_ibm_job(sc: SurfaceCodeCircuit, job_path: str, ratios, seed: int,
                  dt: int, k: int, batch_size: int, device=None, simulator: bool = False):
    """
    Parse an IBM job once and return one IBMJobDecoder per split ratio, each
    pre-loaded with its slice of detections and logical flips.

    ``ratios`` is an iterable of fractions that need not sum to 1; any
    remainder is discarded. Splits are drawn from a permutation seeded with
    ``seed`` so that multiple runs see the same assignment.
    """
    template = IBMJobDecoder(
        sc, job_path=job_path, simulator=simulator, dt=dt, k=k,
        batch_size=batch_size, device=device,
    )
    template._load_job_data()
    n_total = len(template.logical_flips)
    perm = np.random.RandomState(seed).permutation(n_total)

    splits = []
    offset = 0
    for r in ratios:
        n = int(n_total * r)
        idx = perm[offset:offset + n]
        ds = IBMJobDecoder(
            sc, job_path=job_path, simulator=simulator, dt=dt, k=k,
            batch_size=batch_size, device=device,
        )
        ds.detections = template.detections[idx]
        ds.logical_flips = template.logical_flips[idx]
        splits.append(ds)
        offset += n
    return splits


def evaluate_dataset(model, dataset, n_batches: int = 20) -> float:
    """Average accuracy of ``model`` over ``n_batches`` from ``dataset``."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, ei, lab, lm, ea, flips = dataset.generate_batch()
            out = model.forward(x, ei, ea, lab, lm)
            correct += (torch.round(out) == flips).sum().item()
            total += flips.numel()
    return correct / total if total > 0 else 0.0


def decode(distance: int, T: int, job_path: str, finetuned: bool = False):
    from gru_decoder import GRUDecoder
    from args import Args
    args = Args(
        distance=distance,
        dt=2,
        embedding_features=[5, 32, 64, 128, 256],
        hidden_size=128,
        n_layers=4, 
    )

    model = GRUDecoder(args)
    model_path = f"./models/distance{distance}_ibm.pt" if finetuned else f"./models/distance{distance}.pt"
    ckpt = torch.load(model_path, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    sc = SurfaceCodeCircuit(distance=distance, T=T)
    dataset = IBMJobDecoder(
        sc,
        job_path=job_path,
        dt=args.dt,
        k=args.k,
    )
    x, edge_index, labels, label_map, edge_attr, flips = dataset.generate_batch()

    with torch.no_grad():
        predictions = model.forward(x, edge_index, edge_attr, labels, label_map)

    predicted_flips = torch.round(predictions).int()
    accuracy = (predicted_flips.squeeze() == flips.squeeze()).float().mean()
    print(f"GNN-RNN accuracy on hardware data: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":

    D, T = 3, 20
    JOB = "jobs/job_d3_T20_shots50000_d7fmgem2cugc739qov6g.json"

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

    # GNN-RNN
    # Finetuned = True to use trained model, currently only trained on dist 3.
    decode(
        distance=D,
        T=T,
        job_path=JOB,
        finetuned=False,
    )