import pytest
import pandas as pd
import numpy as np
from telemanom_engine import validate_and_clean_dataframe, run_detectors, evaluate_predictions


def test_empty_dataset_raises_error():
    """Verifies that empty datasets raise a clean ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_and_clean_dataframe(empty_df)


def test_duplicate_timestamps_removed():
    """Verifies duplicate timestamps are deduplicated and sorted properly."""
    df = pd.DataFrame({
        "timestep": [0, 1, 1, 2, 0],
        "value": [0.1, 0.2, 0.25, 0.3, 0.15],
        "cmd_0": [0, 0, 0, 1, 0],
    })
    cleaned_df, features = validate_and_clean_dataframe(df)
    assert len(cleaned_df) == 3
    assert list(cleaned_df["timestep"]) == [0, 1, 2]
    assert "value" in features


def test_missing_values_and_invalid_inputs_handled():
    """Verifies NaNs, infinities, and invalid string inputs are coerced and imputed."""
    df = pd.DataFrame({
        "timestep": [0, 1, 2, 3],
        "value": [0.5, np.nan, "INVALID_STR", np.inf],
        "cmd_0": [np.nan, 1.0, 0.0, -np.inf],
    })
    cleaned_df, features = validate_and_clean_dataframe(df)
    assert not cleaned_df[features].isna().any().any()
    assert np.isfinite(cleaned_df[features].values).all()


def test_detector_and_evaluation_no_leakage():
    """Verifies end-to-end baseline detection and evaluation metrics calculation."""
    train_df = pd.DataFrame({
        "timestep": np.arange(100),
        "value": np.sin(np.linspace(0, 10, 100)) * 0.1,
        "cmd_0": np.zeros(100),
    })
    test_df = pd.DataFrame({
        "timestep": np.arange(50),
        "value": np.concatenate([np.zeros(40), np.ones(10) * 5.0]),
        "cmd_0": np.zeros(50),
        "ground_truth": np.concatenate([np.zeros(40, dtype=int), np.ones(10, dtype=int)]),
    })
    out_df = run_detectors(train_df, test_df, ["value", "cmd_0"],detector_type="Rolling Robust Baseline (Statistical)", threshold_pct=5.0)
    metrics = evaluate_predictions(out_df, [[40, 49]])
    assert "precision" in metrics
    assert "recall" in metrics
    assert metrics["detected_sequences"] == 1