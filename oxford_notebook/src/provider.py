from pathlib import Path
from src.loader import DuckDBLoader, InvalidColumnError, DataNotFoundError
from IPython.display import display, Markdown
import pandas as pd
import atexit

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


# this should be internal, private to the module. Singleton-like global instance. 
loader_global: DuckDBLoader | None = None

_atexit_registed : bool = False

# ----------------------------
# init and helpers
# ----------------------------

# to make it super simple we auto detect the db path 
def _auto_db_path(project_root: Path) -> Path:
    search_dirs = [
        project_root / "data",
        project_root / "backend" / "data",
    ]
    candidates: list[Path] = []
    for d in search_dirs:
        if d.exists():
            candidates.extend(d.glob("*.duckdb"))
    # dedupe + stable order
    candidates = sorted({p.resolve() for p in candidates})

    if not candidates:
        raise FileNotFoundError(
            "No .duckdb file found. Put a DuckDB file into /data (recommended) "
            "or /backend/data, or pass db_path explicitly to provider.init(db_path=...)."
        )

    # pick newest if multiple
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def _safe_close_at_exit() -> None:
    try:
        close()
    except Exception:
        pass

def init(db_path: Path | None = None, table_name: str ="my_table", project_root: Path | None = None, read_only: bool = True) -> None:
    global loader_global, _atexit_registed
    
    if not _atexit_registed:
        atexit.register(_safe_close_at_exit)
        _atexit_registed = True
        
    if project_root is None:
        here = Path.cwd().resolve()
        project_root = next((p for p in [here, *here.parents] if (p / "data").exists()), here)
    
    if db_path is None:
        db_path = _auto_db_path(project_root) 
        
    # Autodetect table name if not provided or set to None/empty
    autodetect_table = table_name is None or table_name == "" or table_name == "my_table"
    if autodetect_table:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=read_only)
        tables = con.execute("SHOW TABLES").fetchall()
        if not tables:
            raise RuntimeError(f"No tables found in DuckDB database: {db_path}")
        table_name = tables[0][0]
        con.close()
           
    loader_global = DuckDBLoader(Path(db_path), table_name=table_name, read_only=read_only)
    
    # proof DB is loaded and ready (connection + table readable)
    loader_global.con.execute("SELECT 1").fetchone()
    loader_global.con.execute(f"SELECT 1 FROM {table_name} LIMIT 1").fetchone()
    print(f"[✓] DuckDB ready | Database: {Path(db_path).name} | Table: {table_name}") 
    
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

def _show_user_error(message: str, hint: str | None = None) -> None:
    text = f"###\n**{message}**"
    if hint:
        text += f"\n\n *Hint:* {hint}"
    display(Markdown(text))
    
def _client_call(fn, *, empty_return, hint: str | None = None):
    if loader_global is None:
        raise RuntimeError("provider.init() must be called first")

    try:
        return fn()
    except (DataNotFoundError, InvalidColumnError) as e:
        _show_user_error(str(e), hint=hint)
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

def slots(plate: str) -> list[float]:
    plate = str(plate)
    return _client_call(
        lambda: loader_global.list_slots_for_plate(plate),
        empty_return=[],
        hint="Try provider.plates() to see valid plates.",
    )
    
def plate_slots() -> list[tuple[str, float]]:
    """
    Returns all distinct (plate, slot) pairs in the DB.
    """
    return _client_call(
        lambda: loader_global.list_plate_slots_flat(),
        empty_return=[],
        hint="Check provider.init() and provider.where() to see if the Database is connected.",
    )
    
def signals(plate: str, slot: float | None = None, data_origin: str | None = None) -> pd.DataFrame:
    plate = str(plate)
    return _client_call(
        lambda: loader_global.list_signals(plate, slot, data_origin=data_origin),
        empty_return=[],
        hint="Try provider.slots(<plate>) or list all plates and its slots with provider.plate_slots().",
    )

def data_origin(plate: str, slot: float | None = None) -> list[str]:
    plate = str(plate)
    return _client_call(
        lambda: loader_global.list_data_origins(plate, slot),
        empty_return=[],
        hint="Try provider.plate_slots() to list all slots and plates.",
    )

def df(plate: str, slot: float | None = None, **kwargs) -> pd.DataFrame:
    plate = str(plate) 
    return _client_call(
        lambda: loader_global.get_data_df(plate, slot=slot, **kwargs),
        empty_return=pd.DataFrame(),
        hint="Check plate/slot and filters. Use provider.schema() to see valid columns. and provider.plate_slots() to list all available slots for plates",
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
    



