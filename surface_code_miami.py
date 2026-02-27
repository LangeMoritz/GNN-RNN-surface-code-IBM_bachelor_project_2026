from qiskit import transpile, ClassicalRegister, QuantumCircuit, QuantumRegister, AncillaRegister
from collections import defaultdict
import math
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeDecoder, RuntimeEncoder, SamplerV2 as Sampler
import os
import numpy as np
from collections import Counter
#from dotenv import load_dotenv
import matplotlib as plt

# load_dotenv()

# QiskitRuntimeService.save_account(
# token=os.getenv("IBM_OPEN_KEY"), # Use the 44-character API_KEY you created and saved from the IBM Quantum Platform Home dashboard
# )

class SurfaceCodeCircuit:

    def __init__ (
            self, 
            distance: int, 
            xbasis: bool = False):
    


        super().__init__()
        
        self.distance = distance
        self.num_dqubits = distance ** 2
        self.num_aqubits = distance **2 -1
        self._xbasis = xbasis
        
        self.code_bit = ClassicalRegister(self.num_dqubits, "code_bit")
        self.code_qubit = QuantumRegister((self.num_dqubits), "code_qubit")
        self.measure_qubit = AncillaRegister((self.num_aqubits), "measure_qubit")
        self.qubit_registers = {"code_qubit", "measure_qubit"}
        
        self.measure_bits = []

        self.circuit = QuantumCircuit(self.code_qubit, self.measure_qubit)
        self.circuit.cx(self.code_qubit[0], self.code_qubit[1])
        self.circuit.cx(self.code_qubit[0], self.code_qubit[3])
        self.circuit.cx(self.code_qubit[1], self.code_qubit[2])
        self.circuit.cx(self.code_qubit[1], self.code_qubit[4])

        
    

    #wdef syndrome_measurement(self):
             


test = SurfaceCodeCircuit(5)
test.circuit.draw()