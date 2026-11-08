from __future__ import annotations

import pytest
from api.main import load_artifacts
from api.schemas import PredictionRequest


def test_model_prediction_regression():
    """Verify that model loading and inference produce deterministic outputs."""
    import pandas as pd
    artifacts = load_artifacts()
    clf = artifacts["classifier"]
    reg = artifacts["risk_model"]
    le = artifacts["label_encoder"]
    features = artifacts["features"]

    # Test Case 1: Standard Rice profile
    row1 = pd.DataFrame([[90.0, 42.0, 43.0, 25.0, 80.0, 6.5, 200.0]], columns=features)
    pred_idx1 = clf.predict(row1)[0]
    crop1 = le.inverse_transform([pred_idx1])[0]
    risk1 = reg.predict(row1)[0]

    assert crop1 == "rice"
    assert 0.0 <= risk1 <= 100.0

    # Test Case 2: Standard Apple profile
    row2 = pd.DataFrame([[20.0, 130.0, 200.0, 22.0, 92.0, 6.0, 110.0]], columns=features)
    pred_idx2 = clf.predict(row2)[0]
    crop2 = le.inverse_transform([pred_idx2])[0]
    risk2 = reg.predict(row2)[0]

    assert crop2 == "apple"
    assert 0.0 <= risk2 <= 100.0

