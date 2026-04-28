from qiskit import transpile, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import Layout
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder
from qiskit_ibm_runtime import SamplerV2 as Sampler
import json
import os
from dotenv import load_dotenv

# Physical grid directions (qubit number = row*10 + col)
W, N, S, E = -1, -10, +10, +1

# X-type stabilizer uses Z pattern: NW, NE, SW, SE -> W, N, S, E on chip
# Z-type stabilizer uses N pattern: NW, SW, NE, SE -> W, S, N, E on chip
X_ORDER = [W, N, S, E]
Z_ORDER = [W, S, N, E]

CHIP_MAP = {
    3: {
        "data": [32, 23, 14, 43, 34, 25, 54, 45, 36],
        "ancilla": [13, 42, 33, 24, 44, 35, 26, 55],
        "x_type": {0, 2, 5, 7},
    },
     5: {
        "data": [
            41, 32, 23, 14, 5,
            52, 43, 34, 25, 16,
            63, 54, 45, 36, 27,
            74, 65, 56, 47, 38,
            85, 76, 67, 58, 49,
        ],
        "ancilla": [
            22, 4,
            51, 42, 33, 24, 15,
            53, 44, 35, 26, 17,
            73, 64, 55, 46, 37,
            75, 66, 57, 48, 39,
            86, 68,
        ],
        "x_type": {0, 1, 3, 5, 8, 10, 13, 15, 18, 20, 22, 23},
    },
}
#         5: {
#             "data": [
#                 51, 42, 33, 24, 15,
#                 62, 53, 44, 35, 26,
#                 73, 64, 55, 46, 37,
#                 84, 75, 66, 57, 48,
#                 95, 86, 77, 68, 59,
#             ],
#             "ancilla": [
#                 41, 23,
#                 72, 52, 43, 34, 25,
#                 63, 54, 45, 36, 16,
#                 94, 74, 65, 56, 47,
#                 85, 76, 67, 58, 38,
#                 87, 69,
#             ],
#             "x_type": {0, 1, 4, 6, 7, 9, 14, 16, 17, 19, 22, 23},
#         },
# }
   

class SurfaceCodeCircuit:

    def __init__(self, distance: int, T: int, xbasis: bool = False):
        self.distance = distance
        self.T = 0
        self._xbasis = xbasis

        layout = CHIP_MAP[distance]
        self.data_physical = layout["data"]
        self.ancilla_physical = layout["ancilla"]
        self.x_type = layout["x_type"]

        self.code_qubit = QuantumRegister(len(self.data_physical), "code_qubit")
        self.measure_qubit = QuantumRegister(len(self.ancilla_physical), "measure_qubit")
        self.code_bit = ClassicalRegister(len(self.data_physical), "code_bit")
        self.measure_bits = []

        self.circuit = QuantumCircuit(self.code_qubit, self.measure_qubit)

        self.data_idx = {phys: i for i, phys in enumerate(self.data_physical)}
        self.data_set = set(self.data_physical)

        # Per-ancilla list of data-qubit indices that the stabilizer acts on.
        # Used by IBMJobDecoder
        self.stabilizer_data = {}
        for anc_i, anc_p in enumerate(self.ancilla_physical):
            order = X_ORDER if anc_i in self.x_type else Z_ORDER
            neighbors = []
            for direction in order:
                nb = anc_p + direction
                if nb in self.data_set:
                    neighbors.append(self.data_idx[nb])
            self.stabilizer_data[anc_i] = neighbors

        # Default: False
        if self._xbasis:
            self.circuit.h(self.code_qubit)

        for _ in range(T):
            self.syndrome_measurement()
        if T != 0:
            self.readout()

    def make_layout(self):
        layout_map = {}
        for qubit, phys in zip(self.code_qubit, self.data_physical):
            layout_map[qubit] = phys
        for qubit, phys in zip(self.measure_qubit, self.ancilla_physical):
            layout_map[qubit] = phys
        return Layout(layout_map)

    def syndrome_measurement(self):
        num_ancilla = len(self.ancilla_physical)
        mbit = ClassicalRegister(num_ancilla, f"round_{self.T}_measure_bit")
        self.measure_bits.append(mbit)
        self.circuit.add_register(mbit)

        # H on X-type ancillas
        for i in range(num_ancilla):
            if i in self.x_type:
                self.circuit.h(self.measure_qubit[i])

        self.circuit.barrier()
        for step in range(4):
            for anc_i, anc_p in enumerate(self.ancilla_physical):
                is_x = anc_i in self.x_type
                direction = X_ORDER[step] if is_x else Z_ORDER[step]
                neighbour = anc_p + direction
                if neighbour in self.data_set:
                    dat_i = self.data_idx[neighbour]
                    if is_x:
                        self.circuit.cx(self.measure_qubit[anc_i], self.code_qubit[dat_i])
                    else:
                        self.circuit.cx(self.code_qubit[dat_i], self.measure_qubit[anc_i])

        self.circuit.barrier()
        # H on X-type ancillas
        for i in range(num_ancilla):
            if i in self.x_type:
                self.circuit.h(self.measure_qubit[i])

        # Measure all ancillas
        for j in range(num_ancilla):
            self.circuit.measure(self.measure_qubit[j], self.measure_bits[self.T][j])

        # self.circuit.barrier()
        self.T += 1

    def readout(self):
        """
        Readout of all code qubits, which corresponds to a logical measurement
        as well as allowing for a measurement of the syndrome to be inferred.
        """
        if self._xbasis:
            self.circuit.h(self.code_qubit)
        self.circuit.add_register(self.code_bit)
        self.circuit.measure(self.code_qubit, self.code_bit)


def job_path(job_id, distance, T, shots):
    return os.path.join("jobs", f"d{distance}_T{T}_shots{shots}_{job_id}.json")


def get_runtime_service():
    load_dotenv(dotenv_path="notes/.env")
    return QiskitRuntimeService(token=os.getenv("IBM_KEY"), instance="Surface Codes - Bachelor Thesis 2")


def submit_to_ibm(distance: int, T: int, shots: int):
    """
    Build circuit, transpile, and submit to IBM hardware.
    """
    service = get_runtime_service()
    backend = service.backend("ibm_miami")

    sc = SurfaceCodeCircuit(distance=distance, T=T)
    transpiled = transpile(
        sc.circuit,
        backend=backend,
        initial_layout=sc.make_layout(),
        optimization_level=2,
        seed_transpiler=42,
    )

    sampler = Sampler(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    print(f"Job submitted: {job.job_id()} | d={distance}, T={T}, shots={shots}")
    return job


def save_job_result(job_id: str, distance: int, T: int, shots: int):
    """
    Retrieve a completed IBM job and save results to JSON.
    """
    service = get_runtime_service()
    result = service.job(job_id).result()

    path = job_path(job_id, distance, T, shots)
    with open(path, "w") as f:
        json.dump(result, f, cls=RuntimeEncoder)
    print(f"Results saved to {path}")
    return path


# Adjust params here, uncomment one step at a time.
if __name__ == "__main__":
    D, T, SHOTS = 3, 10, 1000
    # 1: Submit to IBM
    #submit_to_ibm(distance=D, T=T, shots=SHOTS)

    # 2: After job completes, save results (paste your job ID)
    JOB = "d7o6gck97osc73dsrppg"
    #save_job_result(JOB, distance=D, T=T, shots=SHOTS)

    # 3: Decode with GNN-RNN


