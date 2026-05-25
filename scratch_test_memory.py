import sys
from pathlib import Path
import tracemalloc
import numpy as np

sys.path.append(str(Path("src")))
from fh_crowding import binary, ternary

tracemalloc.start()

# Dummy experimental data
T_array = np.array([298]*5)
phiC_array = np.array([0.01, 0.02, 0.05, 0.1, 0.15])
G_array = np.array([-1.0, -1.5, -2.0, -2.5, -3.0])

protein = binary.Protein(SASA=np.array([3000.0]))
cosolute = binary.Cosolute(nu=3.0, chi=0.5, chiTS=0.0)

model = binary.BinaryCrowdingModel(
    protein=protein,
    cosolute=cosolute,
    phiC_min=0.001, phiC_max=0.15, dphiC=0.001
)

print("Memory before fit: ", tracemalloc.get_traced_memory()[0]/1e6, "MB")
model.fit_eps(phiC_array, G_array)
print("Memory after fit: ", tracemalloc.get_traced_memory()[1]/1e6, "MB")

tracemalloc.stop()
