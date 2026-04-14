from qiskit import transpile, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import Layout
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder
from qiskit_ibm_runtime import SamplerV2 as Sampler
import json
import os
from bit_visualization import generate_chip_map
from dotenv import load_dotenv

# Physical grid directions (qubit number = row*10 + col)
W, N, S, E = -1, -10, +10, +1

# X-type stabilizer uses Z pattern: NW, NE, SW, SE -> W, N, S, E on chip
# Z-type stabilizer uses N pattern: NW, SW, NE, SE -> W, S, N, E on chip
X_ORDER = [W, N, S, E]
Z_ORDER = [W, S, N, E]

class SurfaceCodeCircuit:

    def __init__(self, distance: int, T: int, corner_qubit: int, xbasis: bool = False):
        self.distance = distance
        self.T = 0
        self._xbasis = xbasis
        self.corner_qubit = corner_qubit

        layout = generate_chip_map(distance=distance, corner_qubit=corner_qubit, x_max=9, y_max=11, visualization=False)
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


def job_path(job_id, distance, T, shots, corner_qubit):
    return os.path.join("jobs", f"job_d{distance}_T{T}_corner{corner_qubit}_shots{shots}_{job_id}.json")


def get_runtime_service():
    load_dotenv(dotenv_path="notes/.env")
    return QiskitRuntimeService(token=os.getenv("IBM_KEY"), instance="Surface Codes - Bachelor Thesis 2")


def submit_to_ibm(distance: int, T: int, shots: int, corner_qubit: int):
    """
    Build circuit, transpile, and submit to IBM hardware.
    """
    service = get_runtime_service()
    backend = service.backend("ibm_miami")

    sc = SurfaceCodeCircuit(distance=distance, T=T, corner_qubit=corner_qubit)
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


def save_job_result(job_id: str, distance: int, T: int, shots: int, corner_qubit: int):
    """
    Retrieve a completed IBM job and save results to JSON.
    """
    service = get_runtime_service()
    result = service.job(job_id).result()

    path = job_path(job_id, distance, T, shots, corner_qubit)
    with open(path, "w") as f:
        json.dump(result, f, cls=RuntimeEncoder)
    print(f"Results saved to {path}")
    return path


# Adjust params here, uncomment one step at a time.
if __name__ == "__main__":
    D, T, SHOTS, CORNER = 3, 5, 100, 15
    # 1: Submit to IBM
    #submit_to_ibm(distance=D, T=T, shots=SHOTS, corner_qubit=CORNER)

    # 2: After job completes, save results (paste your job ID)
    JOB = "d7elgdu5nvhs73a6t9m0"
    #save_job_result(JOB, distance=D, T=T, shots=SHOTS, corner_qubit=CORNER)

    # 3: Decode with GNN-RNN
    #from ibm_decoder import decode
    #decode(distance=D, T=T, job_path=job_path(JOB, D, T, SHOTS, CORNER), CORNER)

