import sys
import numpy as np
import matplotlib
import arviz as az
import pymc as pm

print("--- PYTHON ENVIRONMENT VERIFICATION ---")
print("Python Executable:", sys.executable)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("ArviZ:", az.__version__)
print("PyMC:", pm.__version__)

try:
    import numba
    print("Numba: INSTALLED (JIT backend ready)")
except ImportError:
    print("Numba: NOT INSTALLED (Recommended: pip install numba)")

print("\nEnvironment check passed successfully.")
