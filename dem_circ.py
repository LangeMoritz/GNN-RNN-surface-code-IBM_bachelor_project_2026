from build_dem_from_detection_events import build_dem_from_ibm_detection_events
from ibm_utils import parse_ibm_job
from surface_code_miami import SurfaceCodeCircuit, X_ORDER, Z_ORDER
from torch_geometric.nn.pool import knn_graph
import sys
import numpy as np
import torch


class Datasimulator:

    def __init__(self, sc: SurfaceCodeCircuit, job, norm = torch.inf, k: int = 20, dt: int = 2, batch_size: int=2048, device: torch.device = None,):
        self.job_path = job
        self.t = sc.T
        self.distance = sc.distance
        self.num_ancilla = len(sc.ancilla_physical)
        self.n_databits = self.distance **2
        self._stabilizer_data = {}
        self.x_type = sc.x_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.k = k
        self.dt = dt
        self.norm = norm
        self._dem = None

        for anc_i, anc_p in enumerate(sc.ancilla_physical):
            is_x = anc_i in sc.x_type
            order = X_ORDER if is_x else Z_ORDER
            neighbors = []
            for direction in order:
                nb = anc_p + direction
                if nb in sc.data_set:
                    neighbors.append(sc.data_idx[nb])
            self._stabilizer_data[anc_i] = neighbors
        
        d = sc.distance
        self._detector_coords = np.zeros((self.num_ancilla, 4), dtype=np.float32)
        for i in range(self.num_ancilla):
            neighbors = self._stabilizer_data[i]
            logical_x = np.mean([n % d for n in neighbors]) # avg col idx
            logical_y = np.mean([n // d for n in neighbors]) # avg row idx
            is_x = 1.0 if i in sc.x_type else 0.0
            self._detector_coords[i] = [logical_x, logical_y, is_x, 1.0 - is_x]


    def _load_job_data(self):
        """Parse IBM job JSON into detection events and logical flips.
        
        Z-stabilizers: final_syndrome = parity of data qubits.
          initial 0. 
        X-stabilizers: data qubits are measured in Z-basis.
          - initial = first ancilla measurement 
          - final   = last ancilla measurement   
          so diff at t=0 is 0
        """
        

        n_data = self.distance ** 2
        final_state, syndromes = parse_ibm_job(
            self.job_path, self.t, n_data, self.num_ancilla
        )

        number_of_shots = final_state.shape[0]
        
        final_syndrome = np.zeros((number_of_shots, self.num_ancilla), dtype=np.uint8)
        initial = np.zeros((number_of_shots, 1, self.num_ancilla), dtype=np.uint8)
        z_list = []
        for anc_i, data_indices in self._stabilizer_data.items():
            if anc_i in self.x_type:
                initial[:, 0, anc_i] = syndromes[:, 0, anc_i]
                final_syndrome[:, anc_i] = syndromes[:, -1, anc_i]
            else:
                z_list.append(anc_i)
                parity = np.zeros(number_of_shots, dtype=np.uint8)
                for d_i in data_indices:
                    parity ^= final_state[:, d_i]
                final_syndrome[:, anc_i] = parity


        # Detection events = XOR between consecutive syndrome rounds
        # Stack: initial | t rounds | final


        # These maps are used to reorder our order of bits to the same order that stim has, e.g adding right->left translated to left -> right
        new_order = np.array([1,0,6,5,4,3,2,11,10,9,8,7,16,15,14,13,12,21,20,19,18,17,23,22])
        new_order_z = np.array([6,4,2,11,9,7,16,14,12,21,19,17])

        initialdiff = initial.copy()
        initialdiff[:, 0, :] = initialdiff[:, 0, :] ^ syndromes[:, 0, :]
        initialdetections_ = initialdiff[:, 0, new_order_z].astype(bool)
        #print(initialdetections_.shape)

        # Final detection, reshaping and only including the z bits for valid dem input
        final_syndrome_b = final_syndrome[:, np.newaxis, :]
        finaldiff = final_syndrome_b.copy()
        finaldiff[:, 0, :] = finaldiff[:, 0, :] ^ syndromes[:, -1, :]
        finaldetections_ = finaldiff[:, 0, new_order_z].astype(bool)
        #print(finaldetections_.shape)
        
        # Dections in middle of measurements where both x and z types can be compared (no special treatment)
        middlediff = syndromes[:, :-1, :] ^ syndromes[:, 1:, :]
        middledetections_ = middlediff[:, :, new_order].astype(bool)
        
        #print(middledetections_.shape)
        # Reshaping to 2d format that DEM requires.
        self.detections = np.concatenate([initialdetections_.reshape(number_of_shots, -1), 
                                          middledetections_.reshape(number_of_shots, -1),
                                          finaldetections_.reshape(number_of_shots, -1)], 
                                          axis=1).astype(bool)
      
        #print(self.detections.shape)
        # Logical observable: parity of first column of data qubits (Z-basis)
        logical_qubits = list(range(self.distance))
        self.logical_flips = np.zeros(number_of_shots, dtype=np.int32)
        for q in logical_qubits:
            self.logical_flips ^= final_state[:, q].astype(np.int32)
    
        return self.detections
            
            

    
    # building the circuit
    def build_circuit(self):
        if self._dem is None:
            det_data = self._load_job_data()
            self._dem = build_dem_from_ibm_detection_events(
                distance=self.distance,
                rounds=self.t,
                detection_events=det_data,
            )
        return self._dem

    def sample_jobs(self, shots_):
        shots = shots_
        dem = self.build_circuit()
        sampler = dem.compile_sampler()
        det_data, obs_data, err_data = sampler.sample(shots=shots, return_errors=True)
        # Split into 3 parts
        initial = det_data[:, :12]          # (5000, 12)
        middle = det_data[:, 12:108]      # (5000, 216)
        final = det_data[:, 108:]        # (5000, 12)

        
        # Reshape
        initial = initial.reshape(shots, 1, 12)
        middle = middle.reshape(shots, 4, 24)
        final = final.reshape(shots, 1, 12)

        positions = [2,4,6,7,9,11,12,14,16,17,19,21]
        initialdetections_ = np.zeros((shots, 1, 24))
        finaldetections_ = np.zeros((shots, 1, 24))
        initialdetections_[:, 0, positions] = initial[:, 0, :]
        finaldetections_[:, 0, positions] = final[:, 0, :]
        middledetections_ = middle

        self.sample_detections = np.concatenate([initialdetections_, 
                                          middledetections_,
                                          finaldetections_], 
                                          axis=1).astype(bool)
        
        self.sample_logical_flips = obs_data

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

    def generate_batch(self):
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
        self.sample_jobs(5000)

        # Filter shots with at least one detection event
        has_event = np.any(self.sample_detections.reshape(len(self.sample_detections), -1), axis=1)
        det_filtered = self.sample_detections[has_event]
        flips_filtered = self.sample_logical_flips[has_event]

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
        flips = torch.from_numpy(flips_batch[:n]).to(self.device)
        labels = labels.to(self.device)
        label_map = label_map.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        return node_features, edge_index, labels, label_map, edge_attr, flips   



### Temporary implementation TODO: Clean this up

# np.set_printoptions(threshold=sys.maxsize)
# ### Initializing surface code and job path
# sc = SurfaceCodeCircuit(distance=5, T=10)
# job = "fine_tune_jobs/job_d7bol795a5qc73dnpkv0_d5_T10_shots1000.json"
# ### Initializing Datasimulator class
# d1 = Datasimulator(sc, job)

if __name__ == "__main__":
    from args import Args

    DISTANCE = 5
    T = 5

    args = Args(
        distance=DISTANCE,
        dt=2,
        embedding_features=[5, 32, 64, 128, 256],
        hidden_size=128,
        n_layers=4,
    )

    sc = SurfaceCodeCircuit(distance=DISTANCE, T=T)

    # Replace job_path with the actual path to your job file
    dataset = Datasimulator(sc, job="ibm_jobs/job_d76p3fmr8g3s73d90gq0_d5_T5_shots5000.json", dt=args.dt, k=args.k)

    x, edge_index, labels, label_map, edge_attr, flips = dataset.generate_batch()



   
# det_data, obs_data, err_data = sampler.sample(shots=1, return_errors=True)


# print("Detection events:\n", det_data)
# print(det_data.shape)




