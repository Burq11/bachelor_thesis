from pathlib import Path
from typing import Iterable
from src.loader import DuckDBLoader, InvalidColumnError, DataNotFoundError, QueryValidationError, NoSignalBearingChannelsError
from IPython.display import display, Markdown
import pandas as pd
import atexit
import warnings

"""
Purpose
-------
This module acts as a high-level interface for accessing and interacting with the database.
It provides the interface logic, which is then utilized in the accompanying notebook for data analysis and visualization.

Use this provider to:
- create wrappers that can be then called in other places with a simple 
provider.<your_wrapper>(<input>)

Example for NOTEBOOKS! or data_processing.py
-----------------------------------------------
# Import  the module 
from src import provider

# 1. Environment and Provider Initialization
%load_ext autoreload
%autoreload 2
%run ../../config/preamble.py
provider.init()

# Now you can process df as needed in your scripts
Plate_22_slot_1_df = provider.df(22, 1, fields=["Time", "Value"])

"""


# Re-exported so notebook code can catch it as `provider.NoSignalBearingChannelsError`.
__all__ = ["NoSignalBearingChannelsError"]

# this should be internal, private to the module. Singleton-like global instance.
loader_global: DuckDBLoader | None = None

_atexit_registed : bool = False

# ----------------------------
# init and helpers
# ----------------------------

def _resolve_project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    here = Path.cwd().resolve()
    return next((p for p in [here, *here.parents] if (p / "data").exists()), here)


def _db_search_dirs(project_root: Path) -> list[Path]:
    return [
        project_root / "data",
        project_root / "backend" / "data",
    ]


def _find_db_candidates(project_root: Path) -> list[Path]:
    """All discoverable .duckdb files, deduped and ordered newest-first."""
    candidates: list[Path] = []
    for d in _db_search_dirs(project_root):
        if d.exists():
            candidates.extend(d.glob("*.duckdb"))
    # dedupe + newest first
    candidates = sorted({p.resolve() for p in candidates})
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def _auto_db_path(project_root: Path) -> Path:
    candidates = _find_db_candidates(project_root)

    if not candidates:
        raise FileNotFoundError(
            "No .duckdb file found. Put a DuckDB file into /data (recommended) "
            "or /backend/data, or pass db_path explicitly to provider.init(db_path=...)."
        )

    # pick newest if multiple, but warn so the choice is not silent
    if len(candidates) > 1:
        newest = candidates[0]
        all_names = ", ".join(p.name for p in candidates)
        warnings.warn(
            f"Multiple DuckDB databases found ({all_names}). "
            f"Auto-selecting the most recently modified one: {newest.name!r}. "
            "To choose a specific database, pass db_path=... to provider.init() "
            "(see provider.list_databases()).",
            stacklevel=2,
        )
    return candidates[0]

def _safe_close_at_exit() -> None:
    try:
        close()
    except Exception:
        pass

def init(db_path: Path | None = None, table_name: str | None = None, project_root: Path | None = None, read_only: bool = True) -> None:
    global loader_global, _atexit_registed

    if not _atexit_registed:
        atexit.register(_safe_close_at_exit)
        _atexit_registed = True

    project_root = _resolve_project_root(project_root)

    # explicit db_path wins; otherwise auto-detect in the search path
    if db_path is None:
        db_path = _auto_db_path(project_root)

    loader_global = DuckDBLoader(Path(db_path), table_name=table_name, read_only=read_only)

    # proof DB is loaded and ready (connection + table readable)
    loader_global.con.execute("SELECT 1").fetchone()
    loader_global.con.execute(f"SELECT 1 FROM {loader_global.table_name} LIMIT 1").fetchone()
    print(f"[✓] DuckDB ready | Database: {Path(loader_global.db_path).name} | Table: {loader_global.table_name}")
    
def where() -> dict:
    """
    Quick debug helper: shows which DB + table the provider is connected to.
    Returns a dict so it displays nicely in notebooks.
    """
    if loader_global is None:
        return {"initialized": False, "hint": "Call provider.init() first"}
    return {
        "initialized": True,
        "db_path": str(loader_global.db_path),
        "table_name": loader_global.table_name,
        "read_only": getattr(loader_global, "read_only", None),
    }


def list_databases(project_root: Path | None = None) -> list[str]:
    """
    List the DuckDB databases discoverable in the search path (data/ and backend/data/)
    as absolute paths, ordered newest-first. Feed a returned path straight to
    provider.init(db_path=...) to select one.
    """
    root = _resolve_project_root(project_root)
    return [str(p) for p in _find_db_candidates(root)]

def _notify_user(issues: Iterable[tuple[str, Iterable[str]]], generic_hint: str | None = None) -> None:
    text = ""
    for message, hints in issues:
        text += f"### {message}\n"
        for hint in hints or []:
            if hint:
                text += f"\n *Hint:* {hint}\n"
    if generic_hint:
        text += f"\n *Hint:* {generic_hint}\n"
    display(Markdown(text))


def _client_call(fn, *, empty_return, hint: str | None = None):
    if loader_global is None:
        raise RuntimeError("provider.init() must be called first")

    try:
        return fn()
    except QueryValidationError as e:
        _notify_user(e.issues, generic_hint=hint)
        return empty_return
    except (DataNotFoundError, InvalidColumnError) as e:
        _notify_user([(str(e), getattr(e, "hints", []))], generic_hint=hint)
        return empty_return

    
# ----------------------------
# wrappers for notebook
# ----------------------------

def plates():
    return _client_call(
        lambda: loader_global.list_plates(),
        empty_return=[],
        hint="Check provider.init() and provider.where() to see if the Database is connected",
    )

def slots(plate: int) -> list[int]:
    plate = int(plate)
    return _client_call(
        lambda: loader_global.list_slots_for_plate(plate),
        empty_return=[],
        hint="Try provider.plates() to see valid plates.",
    )
    
def plate_slots() -> list[tuple[int, int]]:
    """
    Returns all distinct (plate, slot) pairs in the DB.
    """
    return _client_call(
        lambda: loader_global.list_plate_slots_flat(),
        empty_return=[],
        hint="Check provider.init() and provider.where() to see if the Database is connected.",
    )
    
def signals(plate: int, slot: int | None = None, data_origin: str | None = None) -> list[str]:
    plate = int(plate)
    return _client_call(
        lambda: loader_global.list_signals(plate, slot, data_origin=data_origin),
        empty_return=[],
        hint="Try provider.slots(<plate>) or provider.plate_slots() to see valid plates and slots.",
    )

def data_origin(plate: int, slot: int | None = None) -> list[str]:
    plate = int(plate)
    return _client_call(
        lambda: loader_global.list_data_origins(plate, slot),
        empty_return=[],
        hint="Try provider.plate_slots() to list all slots and plates.",
    )

def df(plate: int, slot: int | None = None, **kwargs) -> pd.DataFrame:
    plate = int(plate)
    return _client_call(
        lambda: loader_global.get_data_df(plate, slot, **kwargs),
        empty_return=pd.DataFrame(),
        hint="Try provider.plate_slots() to see valid plates and their slots",
    )
    
def group_data(group_by: list[str], agg: dict, *args, **kwargs) -> pd.DataFrame:
    """
    Returns a grouped and aggregated DataFrame using loader.get_grouped_data.
    """
    return _client_call(
        lambda: loader_global.get_grouped_data(group_by, agg, *args, **kwargs),
        empty_return=pd.DataFrame(),
        hint="Check group_by columns and aggregation dictionary. Use provider.schema() to see valid columns."
    )
    
def schema() -> pd.DataFrame:
    return _client_call(
        lambda: loader_global.schema(),
        empty_return=pd.DataFrame(),
        hint="Check provider.init() and provider.where() to see if Database is connected.",
    )


def table() -> str:
    """Return the name of the table the provider is connected to."""
    if loader_global is None:
        raise RuntimeError("provider.init() must be called first")
    return loader_global.table_name


def query_df(sql: str, params=()) -> pd.DataFrame:
    """Execute a parameterised query via the loader and return a DataFrame."""
    return _client_call(
        lambda: loader_global.query_df(sql, params),
        empty_return=pd.DataFrame(),
        hint="Check that provider.init() has run and that the SQL and params are valid.",
    )


def query_row(sql: str, params=()):
    """Execute a parameterised query via the loader and return the first row."""
    return _client_call(
        lambda: loader_global.query_row(sql, params),
        empty_return=None,
        hint="Check that provider.init() has run and that the SQL and params are valid.",
    )


def slot_metadata_summary(plate: int, *, data_origin: str | None = None) -> pd.DataFrame:
    """One row per slot with overlay metadata needed by the heatmap plots."""
    plate = int(plate)
    return _client_call(
        lambda: loader_global.slot_metadata_summary(plate, data_origin=data_origin),
        empty_return=pd.DataFrame(),
        hint="Check provider.init() and provider.where() to see if the Database is connected.",
    )


def slot_chatter_cases_summary(plate: int, *, data_origin: str | None = None) -> pd.DataFrame:
    """Long-form chatter boundary summary per slot (no raw per-slot loads)."""
    plate = int(plate)
    return _client_call(
        lambda: loader_global.slot_chatter_cases_summary(plate, data_origin=data_origin),
        empty_return=pd.DataFrame(),
        hint="Check provider.init() and provider.where() to see if the Database is connected.",
    )


def close() -> None:
    """
    Closes the DuckDB connection if it is initialized.
    """
    global loader_global
    if loader_global is not None:
        try:
            loader_global.close()  # Close the DuckDB connection
        except Exception:
            pass
        finally:
            loader_global = None  # Reset the global instance
            
            
# ----------------------------
# place for custom wrappers 
# ----------------------------

def axiswise_plot_df(
    plate: int,
    slot: int | None = None,
    *,
    data_origin: str | None = None,
    signals: list[str] | None = None,
    wcs_min: float | None = None,
    wcs_max: float | None = None,
    order_by: str = "Time",
    limit: int | None = None,
) -> pd.DataFrame:
    plate = int(plate)
    return _client_call(
        lambda: loader_global.get_axiswise_plot_df(
            plate,
            slot,
            data_origin=data_origin,
            signals=signals,
            wcs_min=wcs_min,
            wcs_max=wcs_max,
            order_by=order_by,
            limit=limit,
        ),
        empty_return=pd.DataFrame(),
        hint="Check plate/slot and schema via provider.plates(), provider.slots(), provider.schema().",
    )
    
"""
Wrapper Schema
--------------
When adding a new SQL query to loader.py, ensure it is accessible by creating a corresponding wrapper function in provider.py.
To maintain consistency, all wrappers should follow this structure:

1. Define a function in `provider.py` that calls the appropriate method in `loader.py`.
2. Use `_client_call` to handle database interactions and errors gracefully.
3. Provide meaningful hints for debugging in case of errors.
4. Validate inputs (if necessary) before passing them to the loader.
5. Return the result in a format suitable for notebooks or other modules.
"""
# Example :
#
# def new_wrapper_function(param1: str, param2: float) -> list[str]:
#     param1 = str(param1)        # Ensure the parameter is in the correct format
#     return _client_call(
#         lambda: loader_global.<some_loader_function>(param1, param2),
#         empty_return=[],
#         hint="Provide a helpful hint for debugging, e.g., check valid inputs."
#     )
    



