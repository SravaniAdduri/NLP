"""
Frontend utility functions.
"""

import requests
from typing import Optional


def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage string."""
    return f"{confidence * 100:.1f}%"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def get_label_color(label: str) -> str:
    """Get CSS color for a verification label."""
    colors = {
        "SUPPORTED": "#28a745",
        "CONTRADICTED": "#dc3545",
        "NOT_ENOUGH_EVIDENCE": "#ffc107",
        "VERIFIED": "#28a745",
        "REFUTED": "#dc3545",
        "UNVERIFIABLE": "#ffc107",
    }
    return colors.get(label, "#6c757d")


def get_label_background(label: str) -> str:
    """Get CSS background color for a verification label."""
    backgrounds = {
        "SUPPORTED": "#d4edda",
        "CONTRADICTED": "#f8d7da",
        "NOT_ENOUGH_EVIDENCE": "#fff3cd",
        "VERIFIED": "#d4edda",
        "REFUTED": "#f8d7da",
        "UNVERIFIABLE": "#fff3cd",
    }
    return backgrounds.get(label, "#e2e3e5")
