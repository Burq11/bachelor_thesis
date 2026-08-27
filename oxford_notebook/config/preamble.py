"""Shared notebook preamble: imports, paths and template registration.

Provenance: unchanged from Martin Heper's (M.Sc., IWF, TU Berlin) Parquet-based analysis
system, preserved as received at validation_data_access/legacy/Oxford/config/preamble.py,
apart from six unused imports removed in this thesis.
"""


import sys
import warnings
from pathlib import Path
import pandas as pd
import plotly.io as pio

# === Projekt-Root automatisch setzen ===
project_root = Path(__file__).resolve().parents[1]
globals()["project_root"] = project_root

# === Nur den Projekt-Root in sys.path eintragen ===
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
# === Plotly-Konfiguration ===
pio.templates.default = "plotly_dark"

from viz import IWF_template
IWF_template.register_templates()

# === Anzeigeoptionen & Warnungen ===
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore", message="The frame.append method is deprecated.*")

# === Statusmeldung ===
print(
    f"[✓] Preamble geladen | Projekt: {project_root.name} | Template: {pio.templates.default}"
)
