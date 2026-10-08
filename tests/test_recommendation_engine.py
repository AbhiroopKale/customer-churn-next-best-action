from src import config
from src.recommendation_engine import METHOD, matching_rules, recommend, risk_level, segment


def customer(**kw):
    c = dict(Contract="Month-to-month", MonthlyCharges=95.0, tenure=4, InternetService="Fiber optic",
             TechSupport="No", OnlineSecurity="No", PaymentMethod="Electronic check")
    c.update(kw)
    return c


def test_risk_bands():
    assert risk_level(0.0) == "LOW"
    assert risk_level(config.RISK_LOW_BELOW - 0.001) == "LOW"
    assert risk_level(config.RISK_LOW_BELOW) == "MEDIUM"
    assert risk_level(config.RISK_HIGH_FROM - 0.001) == "MEDIUM"
    assert risk_level(config.RISK_HIGH_FROM) == "HIGH"
    assert risk_level(1.0) == "HIGH"


def test_all_rules_fire_for_worst_case_customer():
    assert matching_rules(customer()) == ["R1", "R2", "R3", "R4", "R5"]


def test_no_rule_fires_for_loyal_customer():
    c = customer(Contract="Two year", MonthlyCharges=40.0, tenure=60, TechSupport="Yes",
                 PaymentMethod="Credit card (automatic)")
    assert matching_rules(c) == []


def test_low_risk_gets_no_retention_action():
    r = recommend(customer(), 0.05)
    assert r["risk_level"] == "LOW" and r["rule_ids"] == [] and r["secondary_actions"] == []


def test_high_risk_default_order_without_explanations():
    r = recommend(customer(), 0.9)
    assert r["risk_level"] == "HIGH" and r["rule_ids"][0] == "R1"


def test_explanation_reorders_rules():
    factors = [{"feature": "MonthlyCharges = 95", "contribution": 0.9},
               {"feature": "Contract = Month-to-month", "contribution": 0.1}]
    assert recommend(customer(), 0.9, factors)["rule_ids"][0] == "R2"


def test_high_risk_without_matching_rules_falls_back_to_outreach():
    c = customer(Contract="Two year", MonthlyCharges=40.0, tenure=60, TechSupport="Yes",
                 PaymentMethod="Credit card (automatic)")
    r = recommend(c, 0.95)
    assert r["rule_ids"] == ["R0"] and "outreach" in r["next_best_action"].lower()


def test_engine_declares_itself_rule_based():
    assert "rule-based" in METHOD and recommend(customer(), 0.9)["method"] == METHOD


def test_revenue_proxy_and_segments():
    assert recommend(customer(MonthlyCharges=100.0), 0.5)["revenue_at_risk_proxy"] == 50.0
    assert segment("HIGH", customer(MonthlyCharges=95.0)) == "High-risk / high-value"
    assert segment("HIGH", customer(MonthlyCharges=30.0)) == "High-risk / lower-value"
    assert segment("LOW", customer(tenure=40)) == "Loyal"
