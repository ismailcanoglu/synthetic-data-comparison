import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_v2_4 import validate_config, model_for, run, check_protocol_match

CONFIG = dict(status="EXPLORATORY", scenarios=["s1"], methods=["baseline", "ctgan"],
              seeds=[10000], ctgan_epochs_by_model={"LR": 100, "RF": 10},
              fit_n=750, validation_n=250, test_n=2000, rf_trees=80)

def test_valid_config():
    assert validate_config(CONFIG)["seeds"] == [10000]

def test_reject_duplicate_seeds():
    with pytest.raises(ValueError, match="unique"):
        validate_config({**CONFIG, "seeds": [1, 1]})

def test_reject_missing_model_epoch():
    with pytest.raises(ValueError, match="LR and RF"):
        validate_config({**CONFIG, "ctgan_epochs_by_model": {"LR": 100}})

def test_frozen_run_requires_protocol(tmp_path):
    cfg = {**CONFIG, "status": "FROZEN_BEFORE_CONFIRMATORY_RUN"}
    with pytest.raises(ValueError, match="requires --protocol"):
        run(cfg, tmp_path / "must_not_start")


def test_frozen_run_rejects_mismatched_protocol():
    cfg = {**CONFIG, "status": "FROZEN_BEFORE_CONFIRMATORY_RUN"}
    protocol = {
        "status": "FROZEN_BEFORE_CONFIRMATORY_RUN",
        "evaluation_seeds": list(range(10000, 10030)),
        "development_seeds": list(range(9100, 9110)),
        "scenarios": ["s1", "s2a", "s2b", "s3a", "s3b"],
        "methods": ["baseline", "class_weight", "smote", "ctgan", "oracle"],
        "models": ["LR", "RF"],
        "primary_metric": "ap",
        "primary_policy": "default",
        "ctgan_epochs_by_model": {"LR": 10, "RF": 10},
        "selection_uses_evaluation_results": False,
    }
    with pytest.raises(ValueError, match="Protocol mismatch"):
        check_protocol_match(cfg, protocol)

def test_model_construction():
    assert model_for("LR", 1, False, 80) is not None
    assert model_for("RF", 1, False, 80).n_estimators == 80
