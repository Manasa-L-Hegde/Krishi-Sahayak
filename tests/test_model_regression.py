from __future__ import annotations

import pytest
from api.main import load_artifacts
from api.schemas import PredictionRequest


def test_model_prediction_regression():
    """Verify that model loading and inference produce exact deterministic outputs under pinned training sklearn version."""
    import pandas as pd
    import sklearn
    assert sklearn.__version__.startswith("1.9."), f"Expected training major.minor 1.9.x, got {sklearn.__version__}"

    artifacts = load_artifacts()
    clf = artifacts["classifier"]
    reg = artifacts["risk_model"]
    le = artifacts["label_encoder"]
    features = artifacts["features"]

    # Test Case 1: Standard benchmark profile (N=90, P=42, K=43, temp=25, hum=80, ph=6.5, rain=200)
    row1 = pd.DataFrame([[90.0, 42.0, 43.0, 25.0, 80.0, 6.5, 200.0]], columns=features)
    pred_idx1 = clf.predict(row1)[0]
    crop1 = le.inverse_transform([pred_idx1])[0]
    proba1 = float(clf.predict_proba(row1)[0][pred_idx1])
    risk1 = float(reg.predict(row1)[0])

    assert crop1 == "rice"
    assert round(proba1, 6) == round(0.5528571428571428, 6)
    assert round(risk1, 4) == round(56.667163680086894, 4)

    # Test Case 2: Standard Apple profile (N=20, P=130, K=200, temp=22, hum=92, ph=6.0, rain=110)
    row2 = pd.DataFrame([[20.0, 130.0, 200.0, 22.0, 92.0, 6.0, 110.0]], columns=features)
    pred_idx2 = clf.predict(row2)[0]
    crop2 = le.inverse_transform([pred_idx2])[0]
    risk2 = float(reg.predict(row2)[0])

    assert crop2 == "apple"
    assert 0.0 <= risk2 <= 100.0


def test_sklearn_version_guard_fails_on_mismatch(tmp_path, monkeypatch):
    """Verifies that loading an artifact stamped with a mismatched sklearn version raises RuntimeError."""
    import joblib
    import api.main

    dummy_path = tmp_path / "mismatched_model.joblib"
    joblib.dump({"sklearn_version": "0.24.2", "features": []}, dummy_path)
    monkeypatch.setattr(api.main, "ARTIFACT_PATH", dummy_path)
    api.main.load_artifacts.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="scikit-learn runtime mismatch"):
            api.main.load_artifacts()
    finally:
        api.main.load_artifacts.cache_clear()


def test_torch_version_guard_fails_on_mismatch(tmp_path, monkeypatch):
    """Verifies that loading a DL artifact stamped with a mismatched PyTorch version raises RuntimeError."""
    import joblib
    import api.main

    dummy_dl_path = tmp_path / "mismatched_dl.joblib"
    joblib.dump({"torch_version": "1.12.0+cpu"}, dummy_dl_path)
    monkeypatch.setattr(api.main, "DL_ARTIFACT_PATH", dummy_dl_path)
    api.main.load_dl_artifacts.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="PyTorch runtime mismatch"):
            api.main.load_dl_artifacts()
    finally:
        api.main.load_dl_artifacts.cache_clear()


