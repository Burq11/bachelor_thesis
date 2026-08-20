# Setup

Requires Python 3.12+.

1. Create a virtual environment:
Only if you want an extra virtual enviremont
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Base path for recordings:

The folder that `selector.py` scans for recordings is set in [selector.py:8](selector.py#L8):

```python
parent_directory = "../recordings"
```

Change this to point at wherever the recordings live on the server.

4. Where the DuckDB file is saved:

`cleaner_controller.py` writes the cleaned data into a DuckDB file at [cleaner_controller.py:18](cleaner_controller.py#L18):

```python
con = duckdb.connect('./duckdb.duckdb')
```

This path is relative to the current working directory the script is run from, so `duckdb.duckdb` ends up next to wherever you invoke `selector.py` from. Change the path here if you want the database written elsewhere.

5. Execute

outside via:
```bash
ssh dev@<ip> "cd IEAP/workspace/cleanup && python3 selector.py <recording_name>"
```

on the server:
```bash
cd /cleanup && python3 selector.py <recording_name>
```
