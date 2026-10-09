"""Waste-sorting model comparison pipeline."""
import os

# FIX: must be set BEFORE TensorFlow is imported anywhere.
# Original: set after `import tensorflow`. The C++ runtime reads this variable when it
# loads, so setting it afterwards silently did nothing and TF's INFO/WARNING spam stayed on.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__version__ = "0.1.0"
