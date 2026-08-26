# Provenance: legacy Parquet-based analysis system

**Author:** Martin Heper (M.Sc.), IWF, TU Berlin

Included in this repository **unmodified**, solely so that the measurements
reported in Validation Chapter can be reproduced. Not part of the contribution of
this thesis.

## Modifications
None applied in place. Adjustments required to run the code on the
benchmark machine are kept as patch files in `validation/patches/` and
applied at run time.

## Expected data layout
These scripts read Parquet files from `data/2025-05_Oxford_und_Rebecka/processed/merge_ext`.
The data is not redistributed with this repository

## Files
legacy
│   └── Oxford
│       ├── config
│       │   └── preamble.py
│       ├── data
│       │   └── 2025-05_Oxford_und_Rebecka
│       │       └── processed
│       │           └── merge_ext/PARQUET FILES
│       ├── environment.yml
│       ├── notebooks
│       │   └── 2025-05_Oxford_und_Rebecka
│       │       ├── Oxford-Copy.ipynb
│       │       ├── Oxford.html
│       │       └── Oxford.ipynb
│       ├── README_Oxford.md
│       ├── results
│       │   └── 2025-05_Oxford_und_Rebecka
│       ├── src
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   └── data_processing.py
│       └── viz
│           ├── __init__.py
│           ├── IWF_template.py
│           ├── visualizer.py
│           └── widgets_digital_twin.py