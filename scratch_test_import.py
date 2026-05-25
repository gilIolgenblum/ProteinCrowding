import tracemalloc
tracemalloc.start()
print("Memory before import: ", tracemalloc.get_traced_memory()[0]/1e6, "MB")

import sys
from pathlib import Path
sys.path.append(str(Path("src")))
import fh_crowding.ternary

print("Memory after import: ", tracemalloc.get_traced_memory()[0]/1e6, "MB")
tracemalloc.stop()
