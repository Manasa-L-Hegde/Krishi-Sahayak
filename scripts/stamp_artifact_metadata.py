"""
stamp_artifact_metadata.py
==========================
Records training-time library versions (scikit-learn and PyTorch) into
model artifact dictionaries so runtime version drift can be strictly checked on startup.
"""

from __future__ import annotations

import os
from pathlib import Path
import joblib
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]

# 1. Stamp models/krishi_sahayak.joblib
ml_path = ROOT / "models" / "krishi_sahayak.joblib"
if ml_path.exists():
    ml_artifact = joblib.load(ml_path)
    ml_artifact["sklearn_version"] = sklearn.__version__
    ml_artifact["torch_version"] = torch.__version__
    joblib.dump(ml_artifact, ml_path)
    print(f"Updated {ml_path.name} with sklearn_version={sklearn.__version__}, torch_version={torch.__version__}")

# 2. Stamp models/dl_forecaster.joblib
dl_path = ROOT / "models" / "dl_forecaster.joblib"
if dl_path.exists():
    dl_artifact = joblib.load(dl_path)
    dl_artifact["sklearn_version"] = sklearn.__version__
    dl_artifact["torch_version"] = torch.__version__
    joblib.dump(dl_artifact, dl_path)
    print(f"Updated {dl_path.name} with sklearn_version={sklearn.__version__}, torch_version={torch.__version__}")

