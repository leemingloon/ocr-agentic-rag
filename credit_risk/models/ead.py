"""
EAD (Exposure at Default) – placeholder for credit risk suite.

EAD is the exposure at the time of default (e.g. outstanding balance + undrawn
commitment factor). For term loans, EAD is often the current or projected
balance; for revolvers, it includes a CCF (credit conversion factor).

Usage:
    from credit_risk.models.ead import get_ead
    ead = get_ead(outstanding_balance=10000, undrawn_commitment=0)
"""

from typing import Optional


def get_ead(
    outstanding_balance: float,
    undrawn_commitment: Optional[float] = None,
    ccf: float = 0.0,
) -> float:
    """
    EAD for ECL = PD × LGD × EAD.

    Args:
        outstanding_balance: Current drawn/outstanding amount.
        undrawn_commitment: Undrawn commitment (e.g. revolver); optional.
        ccf: Credit conversion factor for undrawn (0–1). 0 = term loan.

    Returns:
        EAD = outstanding_balance + ccf * (undrawn_commitment or 0).
    """
    undrawn = undrawn_commitment or 0.0
    return outstanding_balance + ccf * undrawn
