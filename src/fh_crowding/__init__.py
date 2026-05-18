from .constants import Constants
from .protein import Protein
from .cosolute import Cosolute, CosoluteMixture
from .binary import BinaryCrowdingModel
from .ternary import TernaryCrowdingModel
from .plotting import BinaryPlotter, TernaryPlotter

# Aliases for backwards compatibility with notebooks
var = Constants
protein = Protein
cosolute = Cosolute
cosolutes = CosoluteMixture
crowding = BinaryCrowdingModel
crowding_ter = TernaryCrowdingModel

__all__ = [
    "Constants",
    "Protein",
    "Cosolute",
    "CosoluteMixture",
    "BinaryCrowdingModel",
    "TernaryCrowdingModel",
    "BinaryPlotter",
    "TernaryPlotter",
    "var",
    "protein",
    "cosolute",
    "cosolutes",
    "crowding",
    "crowding_ter"
]
