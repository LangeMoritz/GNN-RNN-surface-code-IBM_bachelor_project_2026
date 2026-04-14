import json
from collections import Counter
import numpy as np
from qiskit_ibm_runtime import RuntimeDecoder


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

    