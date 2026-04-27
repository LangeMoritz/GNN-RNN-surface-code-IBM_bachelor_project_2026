import stim
import numpy as np
import torch
from torch_geometric.nn.pool import knn_graph
from args import Args
from data import get_sliding_window, labels_from_batch_chunks


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
        self.sliding = getattr(args, "sliding", True)

        coords = circuit.get_detector_coordinates()
        self._detector_coords = np.array(
            [coords[i] for i in range(len(coords))], dtype=np.float32
        )
        # Match the coordinate convention used in data.py.
        self._detector_coords[:, :2] /= 2.0
        self._detector_is_z = np.asarray(detector_is_z, dtype=np.float32)

    def _sliding_window(self, node_features):
        return get_sliding_window(node_features, self.dt, self.rounds, time_col=2)

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

        batch_labels, det_idx = np.where(det_events)
        coords = self._detector_coords[det_idx]
        is_z = self._detector_is_z[det_idx]
        node_features_np = np.empty((len(det_idx), 5), dtype=np.float32)
        node_features_np[:, :3] = coords
        node_features_np[:, 3] = is_z
        node_features_np[:, 4] = 1.0 - is_z

        if self.sliding:
            split_pts = np.searchsorted(batch_labels, np.arange(1, self.batch_size))
            node_features = np.split(node_features_np, split_pts)
            node_features, chunk_labels = self._sliding_window(node_features)
            batch_labels = np.repeat(
                np.arange(self.batch_size),
                [len(features) for features in node_features],
            )
            node_features_np = np.vstack(node_features).astype(np.float32)
        else:
            chunk_labels = (node_features_np[:, 2] // self.dt).astype(np.int64)
            node_features_np[:, 2] = node_features_np[:, 2] % self.dt

        # Map [batch, chunk] -> label integer
        labels, label_map = labels_from_batch_chunks(batch_labels, chunk_labels)

        # Move to GPU before knn_graph so the graph build runs on device.
        node_features = torch.from_numpy(node_features_np).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)
        label_map = torch.from_numpy(label_map).to(self.device)
        flips = torch.from_numpy(obs_flips[:, :1].astype(np.int32)).to(self.device)

        # Edges
        edge_index = knn_graph(node_features, self.k, batch=labels)
        delta = node_features[edge_index[1]] - node_features[edge_index[0]]
        edge_attr = torch.linalg.norm(delta, ord=self.norm, dim=1)
        edge_attr = 1 / edge_attr ** 2

        return node_features, edge_index, labels, label_map, edge_attr, flips
