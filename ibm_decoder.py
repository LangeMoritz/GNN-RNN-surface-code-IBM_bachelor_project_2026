import os
import numpy as np
import torch
from torch_geometric.nn.pool import knn_graph
from surface_code_miami import SurfaceCodeCircuit, X_ORDER, Z_ORDER
from ibm_utils import parse_ibm_job
import sys


class IBMJobDecoder:
    """
    Loads IBM hardware job results and produces GNN-ready batches
    in the same format as data.py Dataset.generate_batch().
    """

    def __init__(self, sc: SurfaceCodeCircuit, job_path: str,
                 simulator: bool = False, k: int = 20, dt: int = 2,
                 norm: float = torch.inf,
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
        self.device = device or torch.device("cpu")

        # Build stabilizer-to-data-qubit map for final syndrome reconstruction, almost same as in surface_code_miami.py
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

        # Build detector coordinates: logical (x, y) from data qubit neighbors.
        # Each ancilla is placed at the centroid of its data qubits in the d×d grid,
        # e.g. a stabilizer touching columns {1,2} rows {0,1} → (x=1.5, y=0.5).
        d = sc.distance
        self._detector_coords = np.zeros((self.num_ancilla, 4), dtype=np.float32)
        for i in range(self.num_ancilla):
            neighbors = self._stabilizer_data[i]
            logical_x = np.mean([n % d for n in neighbors]) # avg col idx
            logical_y = np.mean([n // d for n in neighbors]) # avg row idx
            is_x = 1.0 if i in sc.x_type else 0.0
            self._detector_coords[i] = [logical_x, logical_y, is_x, 1.0 - is_x]

    def _load_job_data(self):
        """
        Parse IBM job JSON into detection events and logical flips.
        """
        n_data = self.distance ** 2
        final_state, syndromes = parse_ibm_job(
            self.job_path, self.t, n_data, self.num_ancilla, self.simulator
        )

        actual_shots = final_state.shape[0]
        
        initial = np.zeros((actual_shots, 1, self.num_ancilla), dtype=np.uint8)
        final_syndrome = np.zeros((actual_shots, self.num_ancilla), dtype=np.uint8)
        
        for anc_i, data_indices in self._stabilizer_data.items():
            # X-type, use first and last data qubit measurements
            if anc_i in self.x_type:    
                initial[:, 0, anc_i] = syndromes[:, 0, anc_i]
                final_syndrome[:, anc_i] = syndromes[:, -1, anc_i]
            else:
                # Z-type, XOR together all data qubit readout values that are neighbour of that Z-stabilizer
                parity = np.zeros(actual_shots, dtype=np.uint8)
                for d_i in data_indices:
                    parity ^= final_state[:, d_i]
                final_syndrome[:, anc_i] = parity
        # Detection events = XOR between consecutive syndrome rounds
        # Stack: initial | t rounds | final
        all_syndromes = np.concatenate(
            [initial, syndromes, final_syndrome[:, np.newaxis, :]], axis=1
        )

        # current col xor previous col
        self.detections = (all_syndromes[:, 1:, :] != all_syndromes[:, :-1, :])
        #self.detections = np.diff(all_syndromes, axis=1).astype(bool)

        # Logical observable: parity of first column of data qubits (Z-basis)
        # !!!!!!! logical_qubits = list(range(0, self.distance ** 2, self.distance))  # [0, 3, 6]
        logical_qubits = list(range(self.distance))  # [0, 1, 2] for d=3
        self.logical_flips = np.zeros(actual_shots, dtype=np.int32)
        for q in logical_qubits:
            # logical_flips = XOR of those logical qubits. 
            # If 1 a logical error occurred. The label that the GNN-RNN tries to predict
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
        # Compute edges.
        edge_index = knn_graph(node_features, self.k, batch=labels)
        delta = node_features[edge_index[1]] - node_features[edge_index[0]]

        # Compute the distances between the nodes:
        edge_attr = torch.linalg.norm(delta, ord=self.norm, dim=1)

        # Inverse square of the norm between two nodes.
        edge_attr = 1 / edge_attr ** 2
        return edge_index, edge_attr

    def generate_batch(self, batch_size=2048):
        """
        Generates a batch of graphs. 

        Returns: 
            node_features: tensor of shape [n, 5] ([x, y, t, (stabilizer type)]).
            edge_index: tensor of shape [n_edges, 2]. Represents the edges, 
                i.e. the adjacency matrix. 
            labels: tensor of shape [n]. Represents which node features belong
                to which combination of batch element and chunk. 
                This is used when computing global_mean_pool following
                graph convolutions. The reason being there is no 
                explicit batch dimension. Therefore, a list of 
                labels is needed to keep track of which node features
                belong to which batch element. Further, each batch element
                consists of multiple graphs, or chunks. Therefore, an integer
                is assigned to each combination of batch element and chunk.
            label_map: tensor of shape [n_graphs]. Maps labels to
                [batch element, chunk].  
            edge_attr: tensor of shape [n_edges]. Represents the edge weights. 
            flips: tensor of shape [batch size]. Indicates if a logical 
                bit- or phase-flip has occured. 1 if it has, 0 otherwise. 
        """
        self._load_job_data()
        

        # Filter shots with at least one detection event
        has_event = np.any(self.detections.reshape(len(self.detections), -1), axis=1)
        det_filtered = self.detections[has_event]
        flips_filtered = self.logical_flips[has_event]

        n = min(batch_size, len(det_filtered))
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
        flips = torch.from_numpy(flips_batch[:n]).to(self.device)
        labels = labels.to(self.device)
        label_map = label_map.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        return node_features, edge_index, labels, label_map, edge_attr, flips


def decode(distance: int, T: int, job_path: str):
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
    model.load_state_dict(torch.load(f"./models/distance{distance}.pt", weights_only=True))
    model.eval()

    sc = SurfaceCodeCircuit(distance=distance, T=T)
    dataset = IBMJobDecoder(sc, job_path=job_path, dt=args.dt, k=args.k)
    x, edge_index, labels, label_map, edge_attr, flips = dataset.generate_batch()

    with torch.no_grad():
        predictions = model.forward(x, edge_index, edge_attr, labels, label_map)

    predicted_flips = torch.round(predictions).int()
    accuracy = (predicted_flips.squeeze() == flips.squeeze()).float().mean()
    print(f"GNN-RNN accuracy on hardware data: {accuracy:.4f}")
    return accuracy

def decode_mwpm(distance: int, T: int, job_path: str):
    import pymatching
    from build_dem_from_detection_events import build_dem_from_ibm_detection_events

    sc = SurfaceCodeCircuit(distance=distance, T=T)
    dataset = IBMJobDecoder(sc, job_path=job_path, dt=2, k=20)
    dataset._load_job_data()

    # detections: (shots, T+1, num_ancilla)
    # stim's rotated_memory_z with rounds=T has T*num_ancilla detectors —
    # the final boundary layer (round_T XOR final_syndrome) is encoded in
    # the logical observable, not as detectors, so drop it here.
    det_flat = dataset.detections[:, :T, :].reshape(len(dataset.detections), -1).astype(np.uint8)

    dem = build_dem_from_ibm_detection_events(distance, T, det_flat)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    predictions = matcher.decode_batch(det_flat)

    accuracy = np.mean(predictions == dataset.logical_flips)
    print(f"MWPM accuracy on hardware data: {accuracy:.4f}")
    return accuracy


# if __name__ == "__main__":
#     from surface_code_miami import job_path
#     D, T, SHOTS = 3, 2, 50
#     JOB = "d72khp1amkec73a1djdg"
#     decode(distance=D, T=T, job_path=job_path(JOB, D, T, SHOTS))

if __name__ == "__main__":
    from surface_code_miami import SurfaceCodeCircuit, job_path

    D, T, SHOTS = 3, 2, 50
    JOB = "d72khp1amkec73a1djdg"

    sc = SurfaceCodeCircuit(distance=D, T=T)
    dataset = IBMJobDecoder(sc, job_path=job_path(JOB, D, T, SHOTS), dt=2, k=20)
    dataset._load_job_data()

    print(f"Raw logical flip rate: {dataset.logical_flips.mean():.3f}")

    n_correct_trivial = (dataset.logical_flips == 0).sum()
    print(f"Trivial decoder accuracy: {n_correct_trivial / len(dataset.logical_flips):.3f}")

    det_flat = dataset.detections.reshape(len(dataset.detections), -1)
    print(f"Shots with any detection: {np.any(det_flat, axis=1).mean():.3f}")
    print(f"Mean detectors fired: {det_flat.sum(axis=1).mean():.1f}")

    for r in range(dataset.detections.shape[1]):
        print(f"Round {r}: {dataset.detections[:, r, :].mean(axis=0).round(3)}")

    print()
    decode_mwpm(distance=D, T=T, job_path=job_path(JOB, D, T, SHOTS))