"""Quick visualization of syndrome data from an IBM hardware job."""
import numpy as np
import matplotlib.pyplot as plt
from surface_code_miami import SurfaceCodeCircuit, X_ORDER, Z_ORDER
from ibm_utils import parse_ibm_job

DISTANCE = 5
T = 10
job_path = "ibm_jobs/job_d6o3ais3pels73a2ah6g.json"

sc = SurfaceCodeCircuit(distance=DISTANCE, T=T)
n_data = DISTANCE ** 2
num_ancilla = len(sc.ancilla_physical)

final_state, syndromes = parse_ibm_job(job_path, T, n_data, num_ancilla)

# Build final syndrome (same logic as ibm_decoder.py)
stabilizer_data = {}
for anc_i, anc_p in enumerate(sc.ancilla_physical):
    is_x = anc_i in sc.x_type
    order = X_ORDER if is_x else Z_ORDER
    neighbors = []
    for direction in order:
        nb = anc_p + direction
        if nb in sc.data_set:
            neighbors.append(sc.data_idx[nb])
    stabilizer_data[anc_i] = neighbors

actual_shots = final_state.shape[0]
final_syndrome = np.zeros((actual_shots, num_ancilla), dtype=np.uint8)
initial = np.zeros((actual_shots, 1, num_ancilla), dtype=np.uint8)

for anc_i, data_indices in stabilizer_data.items():
    if anc_i in sc.x_type:
        initial[:, 0, anc_i] = syndromes[:, 0, anc_i]
        final_syndrome[:, anc_i] = syndromes[:, -1, anc_i]
    else:
        parity = np.zeros(actual_shots, dtype=np.uint8)
        for d_i in data_indices:
            parity ^= final_state[:, d_i]
        final_syndrome[:, anc_i] = parity

all_syndromes = np.concatenate(
    [initial, syndromes, final_syndrome[:, np.newaxis, :]], axis=1
)
detections = np.diff(all_syndromes, axis=1).astype(bool)

# Pick a shot with some detection events to visualize
event_counts = detections.reshape(len(detections), -1).sum(axis=1)
shot = np.argmin(np.abs(event_counts - event_counts.mean()))  # closest to average activity

# Logical flip rate
logical_qubits = list(range(0, DISTANCE ** 2, DISTANCE))
logical_flips = np.zeros(actual_shots, dtype=np.int32)
for q in logical_qubits:
    logical_flips ^= final_state[:, q].astype(np.int32)
flip_rate = logical_flips.mean() * 100

ancilla_labels = [f"{i}{'X' if i in sc.x_type else 'Z'}" for i in range(num_ancilla)]
x_indices = np.array(sorted(sc.x_type), dtype=int)
round_labels = ["init"] + [f"r{i}" for i in range(T)] + ["final"]
detection_labels = ["init->r0"] + [f"r{i}->r{i+1}" for i in range(T - 1)] + [f"r{T-1}->final"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1) All syndrome rounds for one shot
ax = axes[0]
raw_plot = all_syndromes[shot].T.astype(np.float32)
raw_invalid_boundary = np.zeros_like(raw_plot, dtype=bool)
raw_invalid_boundary[x_indices, 0] = True
raw_invalid_boundary[x_indices, -1] = True
raw_plot[raw_invalid_boundary] = np.nan

raw_cmap = plt.get_cmap("Greys").copy()
raw_cmap.set_bad(color="gray", alpha=0.65)
ax.imshow(raw_plot, aspect="auto", cmap=raw_cmap, interpolation="nearest")
ax.set_xlabel("Syndrome slice (init | rounds | final)")
ax.set_ylabel("Ancilla index")
ax.set_yticks(range(num_ancilla))
ax.set_yticklabels(ancilla_labels, fontsize=6)
ax.set_xticks(range(T + 2))
ax.set_xticklabels(round_labels, rotation=45, ha="right", fontsize=7)
ax.set_title(f"Raw syndromes (shot {shot}, X boundaries masked)")

# 2) Detection events for that shot
ax = axes[1]
# X-type detectors are not present at boundary times t=0 and t=T.
# Mask those cells so the plot reflects valid detector support.
detection_plot = detections[shot].T.astype(np.float32)
invalid_boundary = np.zeros_like(detection_plot, dtype=bool)
invalid_boundary[x_indices, 0] = True
invalid_boundary[x_indices, -1] = True
detection_plot[invalid_boundary] = np.nan

cmap = plt.get_cmap("Reds").copy()
cmap.set_bad(color="gray", alpha=0.65)
ax.imshow(detection_plot, aspect="auto", cmap=cmap, interpolation="nearest")
ax.set_xlabel("Detection round")
ax.set_ylabel("Ancilla index")
ax.set_yticks(range(num_ancilla))
ax.set_yticklabels(ancilla_labels, fontsize=6)
ax.set_xticks(range(T + 1))
ax.set_xticklabels(detection_labels, rotation=45, ha="right", fontsize=6)
ax.set_title(f"Detection events (shot {shot}, X boundaries masked)")

plt.suptitle(f"Distance={DISTANCE}, T={T}, shot {shot} ({int(event_counts[shot])} detection events) | Logical flip rate: {flip_rate:.1f}%")
plt.tight_layout()
plt.savefig("syndrome_plot.png", dpi=150)
plt.close()
print(f"Saved to syndrome_plot.png")
