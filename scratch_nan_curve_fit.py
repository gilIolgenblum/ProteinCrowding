import numpy as np
from scipy.optimize import curve_fit
import tracemalloc

tracemalloc.start()

def pade(xy, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19):
    x, y = xy
    return (a1+a2*x+a3*x**2+a4*x**3+a5*y+a6*y**2+a7*y**3+a8*x*y+a9*(x**2)*y+a10*x*(y**2))/ (1+a11*x+a12*x**2+a13*x**3+a14*y+a15*y**2+a16*y**3+a17*x*y+a18*(x**2)*y+a19*x*(y**2))

x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
z[10] = np.nan # introduce nan

try:
    popt, pcov = curve_fit(pade, (x, y), z)
except Exception as e:
    print("Error:", e)

print("Memory: ", tracemalloc.get_traced_memory()[1]/1e6, "MB")
