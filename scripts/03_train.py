"""
Step 3 - Train the spatio-temporal GNN.
No internet needed. Uses MPS (Apple Silicon), CUDA, or CPU automatically.

    python scripts/03_train.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.train import train

if __name__ == "__main__":
    train()
