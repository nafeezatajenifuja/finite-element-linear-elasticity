"""Common output locations and non-interactive scientific plotting."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save_figure(figure, relative: str) -> str:
    path = ROOT / "figures" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return str(path.relative_to(ROOT))


def save_result(name: str, result: dict) -> None:
    path = ROOT / "results" / (name + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(result, default=convert, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Saved {path.relative_to(ROOT)}")


def options(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--full", action="store_true", help="use the original finest meshes (dense solves can be costly)")
    return parser.parse_args()
