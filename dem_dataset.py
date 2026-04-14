import stim
import numpy as np
import torch
from torch_geometric.nn.pool import knn_graph
from args import Args


class DEMDataset:
    """
    Dataset that samples from a calibrated stim.DetectorErrorModel.
    Produces batches in the same 6-tuple format as Dataset and IBMJobDecoder.
    """

    def __init__(self, args: Args, dem: stim.DetectorErrorModel = None,
                 dem_path: str = None, rounds: int = 10,
                 circuit: stim.Circuit = None, x_type: set = None):
        if dem is None and dem_path is None:
            raise ValueError(
                "Provide either a stim.DetectorErrorModel via dem= "
                "or a path to a .dem file via dem_path=."
            )
        if dem is None:
            dem = stim.DetectorErrorModel.from_file(dem_path)

        self.dem = dem
        self.sampler = dem.compile_sampler()
        self.distance = args.distance
        self.dt = args.dt
        self.k = args.k
        self.batch_size = args.batch_size
        self.norm = getattr(args, "norm", torch.inf)
        self.device = args.device
        self.rounds = rounds

        # Build detector coordinates from the provided or generated circuit.
        if circuit is None:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=self.distance,
                rounds=self.rounds,
                after_clifford_depolarization=1e-3,
            )
            rescale_coords = True
        else:
            rescale_coords = False

        coords = circuit.get_detector_coordinates()
        self._detector_coords = np.array(
            [coords[i] for i in range(len(coords))], dtype=np.float32
        )
        # stim.Circuit.generated uses half-integer coords; custom circuit already uses integer
        if rescale_coords:
            self._detector_coords[:, :2] /= 2

        # Build per-detector is_z flag.
        # When x_type is provided (IBM circuit), use it directly.
        # Otherwise fall back to syndrome mask (stim.Circuit.generated).
        if x_type is not None:
            n_anc = len(x_type) + (self.distance ** 2 - 1 - len(x_type))
            z_type_sorted = sorted(set(range(n_anc)) - x_type)
            n_z = len(z_type_sorted)
            # Detector ordering: round 0 Z-only, rounds 1..T-1 all, final Z-only
            is_z_list = []
            is_z_list.extend([1.0] * n_z)  # round 0: Z-type only
            for _ in range(1, rounds):
                for anc_i in range(n_anc):
                    is_z_list.append(0.0 if anc_i in x_type else 1.0)
            is_z_list.extend([1.0] * n_z)  # final: Z-type only
            self._detector_is_z = np.array(is_z_list, dtype=np.float32)
        else:
            self._detector_is_z = None
            sz = self.distance + 1
            syndrome_x = np.zeros((sz, sz), dtype=np.uint8)
            syndrome_x[::2, 1:sz - 1:2] = 1
            syndrome_x[1::2, 2::2] = 1
            syndrome_z = np.rot90(syndrome_x) * 3
            self.syndrome_mask = syndrome_x + syndrome_z

    def generate_batch(self):
        """
        Sample from the DEM and return the standard 6-tuple:
            (node_features, edge_index, labels, label_map, edge_attr, flips)
        """
        det_events, obs_flips, _ = self.sampler.sample(
            shots=self.batch_size * 2
        )

        # Filter to shots with at least one detection event.
        has_event = det_events.any(axis=1)
        det_events = det_events[has_event]
        obs_flips = obs_flips[has_event]

        # Take exactly batch_size samples (resample if needed).
        while len(det_events) < self.batch_size:
            extra_det, extra_obs, _ = self.sampler.sample(shots=self.batch_size)
            mask = extra_det.any(axis=1)
            det_events = np.concatenate([det_events, extra_det[mask]])
            obs_flips = np.concatenate([obs_flips, extra_obs[mask]])

        det_events = det_events[:self.batch_size]
        obs_flips = obs_flips[:self.batch_size]

        # Build node features for each shot.
        all_nodes = []
        batch_labels = []
        chunk_labels = []

        for b_idx in range(self.batch_size):
            fired = np.where(det_events[b_idx])[0]
            if len(fired) == 0:
                continue

            coords = self._detector_coords[fired]  # [n_fired, 3]
            x, y, t = coords[:, 0], coords[:, 1], coords[:, 2]

            chunks = (t // self.dt).astype(int)
            t_local = t % self.dt

            # Stabilizer type
            if self._detector_is_z is not None:
                is_z = self._detector_is_z[fired]
            else:
                xi = x.astype(int)
                yi = y.astype(int)
                is_z = (self.syndrome_mask[yi, xi] == 3).astype(np.float32)

            node_feat = np.column_stack([
                x, y, t_local, is_z, 1.0 - is_z
            ]).astype(np.float32)

            all_nodes.append(node_feat)
            batch_labels.extend([b_idx] * len(fired))
            chunk_labels.extend(chunks.tolist())

        node_features = torch.from_numpy(np.vstack(all_nodes))
        batch_labels = np.array(batch_labels)
        chunk_labels = np.array(chunk_labels)

        # Map [batch, chunk] -> label integer
        label_map = np.column_stack([batch_labels, chunk_labels])
        label_map, counts = np.unique(label_map, axis=0, return_counts=True)
        labels = np.repeat(np.arange(counts.shape[0]), counts).astype(np.int64)
        label_map = torch.from_numpy(label_map)
        labels = torch.from_numpy(labels)

        # Edges
        edge_index = knn_graph(node_features, self.k, batch=labels)
        delta = node_features[edge_index[1]] - node_features[edge_index[0]]
        edge_attr = torch.linalg.norm(delta, ord=self.norm, dim=1)
        edge_attr = 1 / edge_attr ** 2

        # Move to device
        node_features = node_features.to(self.device)
        flips = torch.from_numpy(obs_flips[:, :1].astype(np.int32)).to(self.device)
        labels = labels.to(self.device)
        label_map = label_map.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        return node_features, edge_index, labels, label_map, edge_attr, flips
