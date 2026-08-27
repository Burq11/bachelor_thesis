# Validation of the data access layer

Compares the DuckDB access layer against the legacy Parquet system on filtering, the
heatmap and the plots.

## Folder

```text
validation_data_access/
├── legacy/Oxford/    the legacy system as received
├── tests/
│   ├── harness.py    everything the notebook calls
│   └── validation.ipynb
└── results/          written by the notebook
```

### `legacy/Oxford/`

Martin Heper's (M.Sc., IWF, TU Berlin) Parquet-based analysis system, included so the
validation can be reproduced. Not part of the contribution of this thesis.

The analysis logic is unchanged. Only the import statements were rewritten, from
`from src...` / `from viz...` to absolute package paths, so this tree and the identically
named packages in `oxford_notebook/` can be imported into one interpreter — 12 statements
in 4 files, listed in `results/legacy_import_rewrite.diff`.

### `tests/harness.py`

Six sections, in the order the notebook uses them:

| Section | Contents |
| --- | --- |
| 1. Cases | the plates, slots and parameters used throughout |
| 2. Stacks | importing the legacy and the new tree into one interpreter |
| 3. Legacy | the legacy pipelines, lifted out of their widget callbacks |
| 4. Compare | order-independent frame comparison |
| 5. Figures | plotly figures reduced to comparable signatures |
| 6. Measure | timing and peak memory, one child process per measurement |

### `tests/validation.ipynb`

Three sections — filtering, heatmap, plots — each checking correctness and then cost.

### `results/`

| File | Contents |
| --- | --- |
| `correctness.csv` | one row per check |
| `correctness_filtering.csv` | per-slot row comparison |
| `correctness_heatmap.csv` | per-plate bin and colour-anchor comparison |
| `correctness_axiswise.csv` | per-slot figure comparison |
| `performance.csv` | timings and peak memory |
| `figures/` | the rendered heatmap, both stacks side by side in one PNG |
| `legacy_import_rewrite.diff` | the import rewrite applied to `legacy/Oxford/` |

## Data

Not redistributed with this repository; the notebook stops with an explanation if it is
absent. Expected layout:

```text
legacy/Oxford/data/2025-05_Oxford_und_Rebecka/processed/merge_ext/Platte_{plate}_Nut_{slot}_ext.parquet
```

| | Plates | DataOrigins | Size |
| --- | --- | --- | --- |
| Legacy Parquet, 139 files | 14, 22, 24, 25, 26, 27, 28 | Oscilloscope, LF_Data, HF_Data | 506 MB |
| `oxford_notebook/data/DBold.duckdb` | 14, 22, 24, 25, 26, 27, 28 | Oscilloscope, LF_Data, HF_Data | 1.8 GB |

The comparison runs against `DBold.duckdb`, the DuckDB build of the same Parquet files.
`DBnew.duckdb` is a later campaign the legacy system never processed, with no
`Oscilloscope` origin, so it has no baseline.

## Running it

```bash
conda run -n chatterdetect jupyter nbconvert --to notebook --execute --inplace \
  validation_data_access/tests/validation.ipynb
```

