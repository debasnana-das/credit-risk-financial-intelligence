from __future__ import annotations


def combine_context(credit_tier: str, eps_growth_pct: float):
    # This is deliberately a scenario/communication layer, not a trained underwriting policy.
    tier_attention = {"P1": "lower", "P2": "standard", "P3": "higher", "P4": "highest"}
    attention = tier_attention.get(credit_tier, "unknown")
    if eps_growth_pct > 5:
        outlook = "positive"
    elif eps_growth_pct < -5:
        outlook = "negative"
    else:
        outlook = "stable"

    if credit_tier in {"P3", "P4"}:
        posture = "Enhanced credit review"
    else:
        posture = "Standard credit review"

    if outlook == "negative":
        posture += " + closer portfolio monitoring"

    return {
        "customer_tier_attention": attention,
        "financial_outlook": outlook,
        "scenario_posture": posture,
        "note": (
            "Financial outlook is contextual and does not override customer-level credit assessment. "
            "This layer is a decision-support scenario rule, not an automated approval policy."
        ),
    }
