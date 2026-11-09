"""Mock-only enrolled-template provider; no real biometric data."""
from copy import deepcopy

DEMO_TEMPLATES = [
    {"person_id": "DEMO001", "template_id": "T001", "model_version": "mock-v0", "embedding": [0.1, 0.2, 0.3], "active": True},
    {"person_id": "DEMO002", "template_id": "T002", "model_version": "mock-v0", "embedding": [0.3, 0.2, 0.1], "active": True},
]

def load_active_templates():
    """Return synthetic templates only; Govind's adapter will replace this."""
    return deepcopy([item for item in DEMO_TEMPLATES if item["active"]])
