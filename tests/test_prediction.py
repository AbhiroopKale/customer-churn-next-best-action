import pytest

from src import config
from src.prediction import normalize_inputs

BASE = dict(gender="Male", SeniorCitizen=0, Partner="No", Dependents="No", tenure=4, PhoneService="Yes",
            MultipleLines="Yes", InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
            DeviceProtection="No", TechSupport="No", StreamingTV="Yes", StreamingMovies="Yes",
            Contract="Month-to-month", PaperlessBilling="Yes", PaymentMethod="Electronic check",
            MonthlyCharges=95.0, TotalCharges=None)


def test_no_internet_forces_addons_to_no_internet_service():
    c = normalize_inputs({**BASE, "InternetService": "No"})
    assert all(c[col] == "No internet service" for col in config.ADDON_COLUMNS)


def test_no_phone_forces_multiple_lines():
    assert normalize_inputs({**BASE, "PhoneService": "No"})["MultipleLines"] == "No phone service"


def test_total_charges_estimated_when_missing():
    assert normalize_inputs(BASE)["TotalCharges"] == pytest.approx(4 * 95.0)


@pytest.mark.skipif(not (config.MODELS / "final_model.joblib").exists(), reason="model not trained")
def test_end_to_end_prediction():
    from src.prediction import predict_customer
    res = predict_customer(BASE)
    assert 0.0 <= res["probability"] <= 1.0
    assert res["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert len(res["factors"]) > 0 and res["recommendation"]["method"]
