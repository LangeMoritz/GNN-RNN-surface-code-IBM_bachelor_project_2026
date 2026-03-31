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

# Physical qubit positions and X-type ancilla indices per distance
CHIP_MAP = {
    3: {
        "data":    [50, 41, 32, 61, 52, 43, 72, 63, 54],
        "ancilla": [51, 42, 62, 53, 40, 64, 33, 71],
        "x_type":  {1, 2, 4, 5},
    },
    5: {
        "data": [
            50, 41, 32, 23, 14,
            61, 52, 43, 34, 25,
            72, 63, 54, 45, 36,
            83, 74, 65, 56, 47,
            94, 85, 76, 67, 58,
        ],
        "ancilla": [
            51, 42, 33, 24, 40,
            22, 62, 53, 44, 35,
            15, 71, 73, 64, 55,
            46, 37, 93, 84, 75,
            66, 57, 86, 68,
        ],
        "x_type": {1, 3, 4, 5, 6, 8, 13, 15, 18, 20, 22, 23}, # TODO
    },
}


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

        # H on X-type ancillas
        self.circuit.barrier()
        for i in range(num_ancilla):
            if i in self.x_type:
                self.circuit.h(self.measure_qubit[i])

        # Measure all ancillas
        self.circuit.barrier()
        for j in range(num_ancilla):
            self.circuit.measure(self.measure_qubit[j], self.measure_bits[self.T][j])

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
    return os.path.join("ibm_jobs", f"job_{job_id}_d{distance}_T{T}_shots{shots}.json")


def get_runtime_service():
    load_dotenv()
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
        optimization_level=1,
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
    D, T, SHOTS = 3, 10, 500

    # 1: Submit to IBM
    #submit_to_ibm(distance=D, T=T, shots=SHOTS)

    # 2: After job completes, save results (paste your job ID)
    JOB = "d7553d5bjrds73ebv3s0"
    save_job_result(JOB, distance=D, T=T, shots=SHOTS)

    # 3: Decode with GNN-RNN
    #from ibm_decoder import decode
    #decode(distance=D, T=T, job_path=job_path(JOB, D, T, SHOTS))
