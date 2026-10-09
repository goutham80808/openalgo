"""Market Pulse configuration. Every threshold lives here or in .env."""

import os


def adt_floor_cr() -> float:
    """Minimum 20-day average daily traded value, in crore rupees."""
    return float(os.environ.get("MARKETPULSE_ADT_FLOOR_CR", "1.0"))
