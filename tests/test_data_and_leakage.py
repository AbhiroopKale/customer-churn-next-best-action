import pytest

from src import config
from src.data_preprocessing import prepare, split
from src.train_models import build_preprocessor

pytestmark = pytest.mark.skipif(not config.RAW_CSV.exists(), reason="dataset not downloaded")


def test_features_exclude_id_and_target():
    assert config.ID_COL not in config.FEATURES and config.TARGET not in config.FEATURES


def test_cleaning_removes_nulls_and_encodes_target():
    df = prepare()
    assert df[config.FEATURES].isna().sum().sum() == 0
    assert set(df[config.TARGET].unique()) == {0, 1}
    assert df[config.ID_COL].is_unique


def test_split_is_disjoint_and_stratified():
    df = prepare()
    X_train, X_test, y_train, y_test = split(df)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_preprocessor_learns_only_from_training_data():
    df = prepare()
    X_train, X_test, _, _ = split(df)
    prep = build_preprocessor().fit(X_train)
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    idx = config.NUMERIC_FEATURES.index("tenure")
    assert scaler.mean_[idx] == pytest.approx(X_train["tenure"].mean())
    assert scaler.mean_[idx] != pytest.approx(df["tenure"].mean())
