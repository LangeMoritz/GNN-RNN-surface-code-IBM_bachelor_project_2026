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

    def __init__(self, args: Args, dem: stim.DetectorErrorModel,
                 rounds: int, circuit: stim.Circuit,
                 detector_is_z: np.ndarray):
        self.dem = dem
        self.sampler = dem.compile_sampler()
        self.distance = args.distance
        self.dt = args.dt
        self.k = args.k
        self.batch_size = args.batch_size
        self.norm = getattr(args, "norm", torch.inf)
        self.device = args.device
        self.rounds = rounds

        coords = circuit.get_detector_coordinates()
        self._detector_coords = np.array(
            [coords[i] for i in range(len(coords))], dtype=np.float32
        )
        # Match the coordinate convention used in data.py.
        self._detector_coords[:, :2] /= 2.0
        self._detector_is_z = np.asarray(detector_is_z, dtype=np.float32)

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

            is_z = self._detector_is_z[fired]

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
