import sys
from pathlib import Path
import tracemalloc
import numpy as np

sys.path.append(str(Path("src")))
from fh_crowding import binary, ternary

tracemalloc.start()

# Dummy experimental data
T_array = np.array([298]*5)
phi2_array = np.array([[0.01, 0.02, 0.05, 0.1, 0.15]])
phi3_array = np.array([[0.01, 0.02, 0.05, 0.1, 0.15]])
G_array = np.array([[-1.0, -1.5, -2.0, -2.5, -3.0]])

protein = binary.Protein(SASA=np.array([3000.0]))
cosolutes = ternary.CosoluteMixture(
    nu2=3.0, chi12=0.5, chiTS12=0.0,
    nu3=2.0, chi13=0.3, chiTS13=0.0,
    chi23=0.1, chiTS23=0.0
)

# Use grid 200x200 like in the app
model = ternary.TernaryCrowdingModel(
    protein=protein,
    cosolutes=cosolutes,
    eps2=-0.1, epsTS2=0.0,
    eps3=-0.2, epsTS3=0.0,
    eps23=-0.05, epsTS23=0.0,
    phi2_min=0.001, phi2_max=0.2, dphi2=0.001,
    phi3_min=0.001, phi3_max=0.2, dphi3=0.001,
)

print("Memory before solve_equil: ", tracemalloc.get_traced_memory()[0]/1e6, "MB")
model.solve_equil(print_msg=False)
print("Memory after solve_equil: ", tracemalloc.get_traced_memory()[1]/1e6, "MB")

print("Memory before to_pandas: ", tracemalloc.get_traced_memory()[0]/1e6, "MB")
model.to_pandas()
print("Memory after to_pandas: ", tracemalloc.get_traced_memory()[1]/1e6, "MB")

print("Memory before fit_eps3: ", tracemalloc.get_traced_memory()[0]/1e6, "MB")
model.fit_eps3(phi2_array, phi3_array, G_array)
print("Memory after fit_eps3: ", tracemalloc.get_traced_memory()[1]/1e6, "MB")

tracemalloc.stop()
