"""Fixed sparse encoder plus additive prototype memory (experimental core, hypothesis H1)."""
from .encoder import SparseExpansionEncoder
from .memory import PrototypeMemory
from .model import SparsePrototypeClassifier

__all__ = ["SparseExpansionEncoder", "PrototypeMemory", "SparsePrototypeClassifier"]
