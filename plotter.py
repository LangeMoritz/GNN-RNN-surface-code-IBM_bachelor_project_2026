from surface_code_miami import SurfaceCodeCircuit
from mwpm_decoder import MWPMDecoder
import numpy as np
import matplotlib.pyplot as plt
from ibm_decoder import decode


D, T = 3, 10
JOB = "jobs/dist3/job_d777qp46ji0c738cgnbg_d3_T10_shots100000.json"

arr_x = np.arange(1, T + 1)
arr_mwpm = np.zeros(T)
arr_ibm = np.zeros(T)


for i in range(1, T+1):
    new_T = i
    sc = SurfaceCodeCircuit(distance=D, T=new_T)
    decoder = MWPMDecoder(sc, job_path=JOB, pij_threshold=0.044)
    p_l, p_l_err = decoder.decode()
    accuracy_mwpm = 1 - p_l
    arr_mwpm[i-1] = accuracy_mwpm
    accuracy_ibm = decode(distance=D,T=T,job_path=JOB, finetuned=False)
    arr_ibm[i-1] = accuracy_ibm


plt.plot(arr_x, arr_mwpm, label='mwpm')
plt.plot(arr_x, arr_ibm, label='ibm_decoder')
plt.legend()
plt.xlabel('T rounds')
plt.ylabel('Accuracy')
plt.show()



