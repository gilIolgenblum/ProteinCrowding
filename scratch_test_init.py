import sys
from pathlib import Path
import tracemalloc
import numpy as np

sys.path.append(str(Path("src")))
from fh_crowding import binary, ternary

tracemalloc.start()

protein = binary.Protein(SASA=np.array([3000.0]))
cosolutes = ternary.CosoluteMixture(
    nu2=3.0, chi12=0.5, chiTS12=0.0,
    nu3=2.0, chi13=0.3, chiTS13=0.0,
    chi23=0.1, chiTS23=0.0
)

print("Memory before model: ", tracemalloc.get_traced_memory()[0]/1e6, "MB")

model = ternary.TernaryCrowdingModel(
    protein=protein,
    cosolutes=cosolutes,
    eps2=-0.1, epsTS2=0.0,
    eps3=-0.2, epsTS3=0.0,
    eps23=-0.05, epsTS23=0.0,
    phi2_min=0.001, phi2_max=0.2, dphi2=0.001,
    phi3_min=0.001, phi3_max=0.2, dphi3=0.001,
)

print("Memory after model: ", tracemalloc.get_traced_memory()[0]/1e6, "MB")
tracemalloc.stop()
