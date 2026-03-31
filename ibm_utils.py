import json
from collections import Counter
import numpy as np
from qiskit_ibm_runtime import RuntimeDecoder


def parse_ibm_job(job_path, t, n_data, n_measures, simulator=False):
    """
    Parse IBM job JSON into final data-qubit states and virtual-reset syndromes.

    Handles both simulator (get_counts) and hardware (per-register bitstrings),
    expands compressed counts, reverses IBM's MSB-first bit order, and applies
    no-reset XOR diffing.

    Returns
    -------
    final_state   : np.ndarray, shape (shots, n_data), dtype uint8
    syndromes     : np.ndarray, shape (shots, t, n_measures), dtype uint8
        XOR-diffed syndromes (no-reset correction applied).
    syndromes_raw : np.ndarray, shape (shots, t, n_measures), dtype uint8
        Raw (undiffed) syndrome measurements. Needed for ancillas
        where diffing is inappropriate (e.g. X-type ancillas that
        receive a Hadamard between rounds, effectively resetting them).
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

    # Reverse bit order: IBM MSB-first -> qubit-0 first
    syndromes_nr = syndromes_nr[:, ::-1]
    final_state = final_state[:, ::-1]

    # Reshape into (shots, t, n_measures)
    syndromes_nr = syndromes_nr.reshape(-1, t, n_measures)

    # Keep a raw copy BEFORE diffing
    syndromes_raw = syndromes_nr.copy()

    # No-reset XOR diffing
    diff = (syndromes_nr[:, 1:, :] != syndromes_nr[:, :-1, :]).astype(np.uint8)
    first = syndromes_nr[:, :1, :]
    syndromes = np.concatenate([first, diff], axis=1)

    return final_state, syndromes, syndromes_raw