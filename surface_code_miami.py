from qiskit import transpile, ClassicalRegister, QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.transpiler import Layout
from collections import defaultdict
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeDecoder, RuntimeEncoder, SamplerV2 as Sampler
import numpy as np
import os
from collections import Counter
from dotenv import load_dotenv
import matplotlib as plt



class SurfaceCodeCircuit:

    def __init__ (
            self, 
            distance: int,
            T: int,
            xbasis: bool = False):
    
        super().__init__()
        
        self.distance = distance
        self.num_dqubits = distance ** 2
        self.num_aqubits = distance ** 2 -1
        self.T = 0
       
        self.code_qubit = QuantumRegister((self.num_dqubits), "code_qubit")
        self.measure_qubit = AncillaRegister((self.num_aqubits), "measure_qubit")
        self.code_bit = ClassicalRegister(self.num_dqubits, "code_bit")
        self.measure_bits = []
        
        self.qubit_registers = {"code_qubit", "measure_qubit"}
        
        self.circuit = QuantumCircuit(self.code_qubit, self.measure_qubit)
        self._xbasis = xbasis
        

        if (self._xbasis):
         self.circuit.h(self.code_qubit)
        
        # Physical Qubits 
        if distance == 3:
            self.data_physical = [50, 41, 32, 61, 52, 43, 72, 63, 54] # fixa så man ser type 2s
            self.ancilla_physical = [51, 42, 62, 53, 40, 64, 33, 71]
        elif distance == 5:
            self.data_physical = [
                50, 41, 32, 23, 14,
                61, 52, 43, 34, 25,
                72, 63, 54, 45, 36,
                83, 74, 65, 56, 47,
                94, 85, 76, 69, 58,
            ]
            self.ancilla_physical = [
                51, 42, 33, 24,
                40, 22,
                62, 53, 44, 35, 15,
                71, 73, 64, 55, 46, 37,
                93, 84, 75, 66, 57,
                86, 68,
            ]
        else:
            raise ValueError(f"Unsupported distance: {distance}")
        
        #Lists for which ancilla qubits and code qubits should be entangled in order
        self.entagle_list_code_3d = [1, 2, 3, 4, 5, 8, 2, 3, 4, 5, 6, 9, 1, 4, 5, 6, 7, 8, 2, 5, 6, 7, 8, 9]
        self.entagle_list_ancilla_3d = [2, 3, 4, 6, 7, 8, 2, 3, 5, 6, 7, 8, 1, 2, 3, 4, 6, 7, 1, 2, 3, 5, 6, 7]
        self.entagle_list_code_5d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 25, 1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 2, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        self.entagle_list_ancilla_5d = [3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]

        for _ in range(T - 1):
            self.syndrome_measurement()

        if T != 0:
            self.syndrome_measurement()
            self.readout()

    def make_layout(self):
        layout_map = {}
        for qubit, phys in zip(self.code_qubit, self.data_physical):
            layout_map[qubit] = phys
        for qubit, phys in zip(self.measure_qubit, self.ancilla_physical):
            layout_map[qubit] = phys
        return Layout(layout_map)


    
    def syndrome_measurement(self):
        self.measure_bits.append(ClassicalRegister(self.num_aqubits, "round_" + str(self.T) + "_measure_bit"))
        self.circuit.add_register(self.measure_bits[-1])

        # Select entanglement lists
        if self.distance == 3:
            code_list = self.entagle_list_code_3d
            ancilla_list = self.entagle_list_ancilla_3d
        elif self.distance == 5:
            code_list = self.entagle_list_code_5d
            ancilla_list = self.entagle_list_ancilla_5d
        else:
            raise ValueError(f"Unsupported distance: {self.distance}")
        
        # Entaglement XBASIS???
        self.circuit.barrier()
        for j in range(len(code_list)):
            c_idx = code_list[j] - 1
            a_idx = ancilla_list[j] - 1
            if ancilla_list[j] % 2 == 0:
                # X-type
                self.circuit.h(self.code_qubit[c_idx])
                self.circuit.cx(self.measure_qubit[a_idx], self.code_qubit[c_idx])
                self.circuit.h(self.code_qubit[c_idx])
            else:
                # Z-type
                self.circuit.cx(self.code_qubit[c_idx], self.measure_qubit[a_idx])
        self.circuit.barrier()

        # Measure
        for j in range(self.num_aqubits):
            self.circuit.measure(self.measure_qubit[j], self.measure_bits[self.T][j])

        self.T += 1

                # Entaglement
        # self.circuit.barrier()
        # for j in range(self.entagle_list_code_5d):
        #     if self.entagle_list_anzilla_5d[j] % == 0: #om jämn, varannan anzilla vill vi entangla i x-bas
        #         self.circuit.h(self.code_qubit)
        #         self.circuit.cx(self.measure_qubit[self.entagle_list_anzilla_5d[j]], self.code_qubit[self.entagle_list_code_5d[j]])
        #         self.circuit.h(self.code_qubit)
        #     else:
        #         self.circuit.cx(self.code_qubit[self.entagle_list_code_5d[j]], self.measure_qubit[self.entagle_list_anzilla_5d[j]])
        #  self.circuit.barrier()
        
         # Z. NW, NE, SW, SE?
         # X ancilla -> NE, NW, SE, SW. ancilla -> data, H before and after
         # Z ancilla -> NE, SE, NW, SW. data -> ancilla

    def readout(self):
        """
        Readout of all code qubits, which corresponds to a logical measurement
        as well as allowing for a measurement of the syndrome to be inferred.
        """
        if self._xbasis:
            self.circuit.h(self.code_qubit)
        self.circuit.add_register(self.code_bit)
        self.circuit.measure(self.code_qubit, self.code_bit)

# test = SurfaceCodeCircuit(5,2)
# print(test.circuit.draw())

        
# test = SurfaceCodeCircuit()

# load_dotenv()

# QiskitRuntimeService.save_account(
# token=os.getenv("IBM_KEY"),
# )

# service = QiskitRuntimeService(token=token, platform="ibm_quantum_service")
# backend_name = "ibm_miami"
# backend = service.backend(backend_name)

# transpiled_circuit = transpile(
#     test.circuit,
#     backend=backend,
#     initial_layout=test.make_layout(test.distance),
#     optimization_level=1,
#     seed_transpiler=42
# )

sc = SurfaceCodeCircuit(distance=3, T=1)
print(sc.circuit.draw(output="text", fold=-1))