"""Options analytics: Black-Scholes, binomial American, SVI surface.

Modules 14-15 of the course. Pure NumPy/SciPy — no QuantLib required.
"""

from .binomial import crr_price, early_exercise_premium
from .black_scholes import Greeks, greeks, implied_vol, price
from .svi import (
    SVIParams,
    fit_svi_slice,
    implied_vol_from_total_variance,
    smile_residuals,
    svi_total_variance,
)

__all__ = [
    "Greeks",
    "SVIParams",
    "crr_price",
    "early_exercise_premium",
    "fit_svi_slice",
    "greeks",
    "implied_vol",
    "implied_vol_from_total_variance",
    "price",
    "smile_residuals",
    "svi_total_variance",
]
