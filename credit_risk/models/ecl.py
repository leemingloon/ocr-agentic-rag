"""
ECL (Expected Credit Loss) – combines PD, LGD, EAD for IFRS 9–style ECL.

ECL = PD × LGD × EAD (simplified one-year; staging and lifetime ECL
require institution-specific logic).

Usage:
    from credit_risk.models.ecl import compute_ecl
    from credit_risk.models.pd_model import PDModel
    from credit_risk.models.lgd_model import LGDModel
    from credit_risk.models.ead import get_ead

    pd_model = PDModel(mode="local")
    pd_model.load("models/pd/pd_model_local_v1.pkl")
    lgd_model = LGDModel(constant_lgd=0.45)
    pd_val = pd_model.predict_pd(features)
    lgd_val = lgd_model.predict_lgd(None)
    ead_val = get_ead(outstanding_balance=10000)
    ecl = compute_ecl(pd_val, lgd_val, ead_val)
"""

from typing import Optional

from .lgd_model import LGDModel
from .ead import get_ead


def compute_ecl(
    pd: float,
    lgd: float,
    ead: float,
) -> float:
    """
    ECL = PD × LGD × EAD (one-period, simplified for portfolio/IFRS 9 context).

    Args:
        pd: Probability of default (0–1).
        lgd: Loss given default (0–1).
        ead: Exposure at default (e.g. currency units).

    Returns:
        Expected credit loss in same units as EAD.
    """
    return pd * lgd * ead
