"""NEXT BEST ACTION ENGINE  --  transparent RULE-BASED / heuristic recommendations.

This is NOT a machine-learning recommender. The dataset contains no history of retention
offers or their outcomes, so nothing here has been learned or proven to reduce churn.
Each rule uses only columns that exist in the Telco data. Rules are hypotheses to be
tested (e.g. with A/B tests) in a real deployment.
"""
from src import config

METHOD = "rule-based heuristic (not learned from intervention outcomes)"

# rule_id -> (action text, model features whose SHAP contribution makes the rule relevant)
RULES = {
    "R1": ("Offer an incentive to move to a 1- or 2-year plan", ["Contract"]),
    "R2": ("Review plan fit and offer a lower-cost / discounted plan",
           ["MonthlyCharges", "charge_category"]),
    "R3": ("Early-retention offer with a welcome / onboarding call", ["tenure", "tenure_group"]),
    "R4": ("Offer a trial of TechSupport / OnlineSecurity add-ons "
           "(a service-adoption gap; this dataset has NO complaint or support-ticket data)",
           ["TechSupport", "OnlineSecurity", "InternetService", "n_addons"]),
    "R5": ("Encourage automatic payment with a small incentive", ["PaymentMethod"]),
}


def risk_level(p: float) -> str:
    if p >= config.RISK_HIGH_FROM:
        return "HIGH"
    if p >= config.RISK_LOW_BELOW:
        return "MEDIUM"
    return "LOW"


def segment(level: str, c: dict) -> str:
    """Rule-based business segments (NOT clustering). 'Value' = monthly charges, a proxy only."""
    if level == "HIGH":
        return ("High-risk / high-value" if c["MonthlyCharges"] >= config.HIGH_CHARGE_MIN
                else "High-risk / lower-value")
    if level == "MEDIUM":
        return "Medium-risk"
    return "Loyal" if c["tenure"] >= config.LOYAL_TENURE_MIN else "Low-risk"


def matching_rules(c: dict) -> list:
    hit = []
    if c["Contract"] == "Month-to-month":
        hit.append("R1")
    if c["MonthlyCharges"] >= config.HIGH_CHARGE_MIN:
        hit.append("R2")
    if c["tenure"] <= config.LOW_TENURE_MAX:
        hit.append("R3")
    if (c["InternetService"] != "No" and c["TechSupport"] == "No"
            and c["OnlineSecurity"] == "No"):
        hit.append("R4")
    if c["PaymentMethod"] == "Electronic check":
        hit.append("R5")
    return hit


def _rank(hit: list, factors) -> list:
    """Order matching rules by how strongly the model's own explanation points at the
    related features (Explain -> Recommend link). Ties keep rule order R1..R5."""
    if factors is None or len(factors) == 0:
        return hit
    scores = {}
    for rid in hit:
        drivers = RULES[rid][1]
        scores[rid] = sum(f["contribution"] for f in factors
                          if f["contribution"] > 0 and f["feature"].split(" = ")[0] in drivers)
    return sorted(hit, key=lambda r: (-scores[r], hit.index(r)))


def recommend(customer: dict, probability: float, factors=None) -> dict:
    """factors: optional list of {'feature','contribution'} dicts from SHAP (top factors)."""
    level = risk_level(probability)
    seg = segment(level, customer)
    proxy = round(float(probability) * float(customer["MonthlyCharges"]), 2)
    ranked = _rank(matching_rules(customer), factors)
    base = {"risk_level": level, "segment": seg, "method": METHOD,
            "revenue_at_risk_proxy": proxy}

    if level == "LOW":
        return {**base, "next_best_action": "No retention action needed; routine engagement",
                "secondary_actions": [], "rule_ids": []}
    if level == "MEDIUM":
        return {**base, "next_best_action": "Proactive check-in / light-touch engagement",
                "secondary_actions": [RULES[r][0] for r in ranked[:2]],
                "rule_ids": ranked[:2]}
    if not ranked:
        return {**base, "next_best_action": "Personalised outreach by the retention team",
                "secondary_actions": [], "rule_ids": ["R0"]}
    return {**base, "next_best_action": RULES[ranked[0]][0],
            "secondary_actions": [RULES[r][0] for r in ranked[1:3]],
            "rule_ids": ranked[:3]}
