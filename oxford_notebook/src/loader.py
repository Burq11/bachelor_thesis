
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import duckdb
import pandas as pd

"""
Purpose
-------
This module provides interface for reading data from
DuckDB. It is limited to basic data access and does not
contain analysis logic.

Use this loader to:
- list available plates, slots, signals, and data origins
- load raw data for a given plate and slot
- apply simple database-side filters (columns, signal names, WCS range, origin)
- create your own sql queries

========IMPORTANT=======
!!After creating a new SQL query in this file, ensure it is accessible by adding a corresponding wrapper function in provider.py. 
This allows the query to be used outside of this file (notebooks, data_processing) !!  

"""

table_var = "my_table"

class DataNotFoundError(ValueError):
    """Raised when requested Plate/Nut does not exist"""
    pass

class InvalidColumnError(ValueError):
    """Raised when requested columns are not part of the table schema"""
    pass

class DuckDBLoader:         
    def __init__(
        self,
        db_path: Path,
        table_name: str = table_var,
        default_limit: int = 20000000,  #change the limits later 
        max_limit:  int = 200000000,    #change the limits later
        read_only: bool = False,
    ):
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.default_limit = default_limit
        self.max_limit = max_limit
        self.con = duckdb.connect(str(self.db_path), read_only= read_only)
        
        self._valid_cols = self._load_schema_cols()
      
        
    # ----------------------------
    # Validation
    # ----------------------------
    def _exists(self, where_sql: str, params: list ) -> bool:
        q = f"SELECT 1 FROM {self.table_name} WHERE {where_sql} LIMIT 1"
        return self.con.execute(q,params).fetchone() is not None
    
    def _ensure_plate_exists(self, plate: str) -> None:
        if not self._exists("Platte = ? AND Platte IS NOT NULL", [plate]):
            raise DataNotFoundError(f"Unknown plate={plate!r}.")
        
    def _ensure_plate_slot_exists(self, plate: str, slot: float) -> None:
            if not self._exists("Platte = ? AND Nut = ?", [plate, slot]):
                if not self._exists("Platte = ?", [plate]):
                    raise DataNotFoundError(f"Unknown plate={plate!r}.")
                raise DataNotFoundError(
                    f"No data for (plate={plate!r}, slot={slot}).")
                
    def _validate_fields(self, fields: Optional[Iterable[str]]) -> Optional[list[str]]:
        """
        Returns:
          - None => use "*"
          - list[str] => validated field list
        """
        if not fields:
            return None
        fields = list(fields)
        unknown = [c for c in fields if c not in self._valid_cols]
        if unknown:
            raise InvalidColumnError(f"Unknown column(s): {unknown}. Use provider.schema() to see valid columns")
        return fields
    
    
    # ----------------------------
    # Utilities
    # ----------------------------
    def schema(self) -> pd.DataFrame:
        rows = self.con.execute(f"DESCRIBE {self.table_name}").fetchdf()
        return rows 
    
    def _load_schema_cols(self) -> set[str]:
        rows = self.con.execute(f"DESCRIBE {self.table_name}").fetchall()
        return {row[0] for row in rows} 
    
    # Run a parameterized SQL query and return a pandas DataFrame 
    def query_df(self, sql: str, params=()) -> pd.DataFrame:
        return self.con.execute(sql, params).df()  
     
    # Fetches data using get_data_df and applies groupby/aggregation.
    def get_grouped_data(self, group_by: list[str], agg: dict, *args, **kwargs) -> pd.DataFrame:
        df = self.get_data_df(*args, **kwargs)
        return df.groupby(group_by).agg(agg).reset_index() 
    
    # ----------------------------
    # Basic SQL queries
    # ----------------------------
    def list_plates(self) -> list[str]:
        query = f"""
            SELECT DISTINCT Platte
            FROM {self.table_name}
            WHERE Platte IS NOT NULL
            ORDER BY Platte
        """
        result_rows = self.con.execute(query).fetchall()
        return [row[0] for row in result_rows]
    
    
    def list_slots(self) -> list[float]:
        query = f"""
            SELECT DISTINCT Nut
            FROM {self.table_name}
            WHERE Nut IS NOT NULL
            ORDER BY Nut
        """
        result_rows = self.con.execute(query).fetchall()
        return [row[0] for row in result_rows]
    
    
    ## maybe we dont need this, it just shows all the slots of plates
    def list_plate_slots_flat(self) -> list[tuple[str, float]]:
        query = f"""
            SELECT DISTINCT Platte, Nut
            FROM {self.table_name}
            WHERE Platte IS NOT NULL
            AND Nut IS NOT NULL
            ORDER BY Platte, Nut
        """
        result_rows = self.con.execute(query).fetchall()
        return [(row[0], row[1]) for row in result_rows]
    
    
    def list_slots_for_plate(self, plate: str) -> list[float]:
        self._ensure_plate_exists(plate)
        query = f"""
            SELECT DISTINCT Nut
            FROM {self.table_name}
            WHERE Platte = ?
            AND Nut IS NOT NULL
            ORDER BY Nut
        """
        result_rows = self.con.execute(query, [plate]).fetchall() 
        return [row[0] for row in result_rows]
    
    
    def list_signals(self, plate: str, slot: Optional[float] = None, data_origin: Optional[str] = None) -> pd.DataFrame: ### here we can make the signals a list and not a dataframe
        # If a slot is provided, ensure that the specific (plate, slot) exists.
        # If no slot is provided, return signals for the whole plate.
        self._ensure_plate_exists(plate)
        query = f"""
            SELECT DISTINCT Signal
            FROM {self.table_name}
            WHERE Platte = ?
            AND Signal IS NOT NULL
        """
        params = [plate]

        if slot is not None:
            self._ensure_plate_slot_exists(plate, slot)
            query += " AND Nut = ?"
            params.append(slot)

        if data_origin:
            query += " AND DataOrigin = ?"
            params.append(data_origin)

        query += " ORDER BY Signal"
        df = self.query_df(query, params)

        return df
    
    def list_data_origins(self, plate: str, slot: Optional[float] = None) -> list[str]:
        # If slot provided, return data origins for that (plate, slot),
        # otherwise return all data origins for the plate.
        self._ensure_plate_exists(plate)
        query = f"""
            SELECT DISTINCT DataOrigin
            FROM {self.table_name}
            WHERE Platte = ?
            AND DataOrigin IS NOT NULL
            ORDER BY DataOrigin
        """
        params: list = [plate]

        if slot is not None:
            self._ensure_plate_slot_exists(plate, slot)
            query = query.replace("WHERE Platte = ?", "WHERE Platte = ?\n            AND Nut = ?")
            params.append(slot)

        rows = self.con.execute(query, params).fetchall()
        return [row[0] for row in rows]
            
    
    def get_data_df(self, plate: str, slot: Optional[float] = None, *,
        fields: Optional[Iterable[str]] = None, data_origin: Optional[str] = None,
        signals: Optional[Iterable[str]] = None, wcs_min: Optional[float] = None, 
        wcs_max: Optional[float] = None, order_by_time: bool = True, limit: Optional[int] = None) -> pd.DataFrame:
        
        self._ensure_plate_exists(plate)
        
        validated_fields = self._validate_fields(fields)
        select_clause = "*" if validated_fields is None else ", ".join(validated_fields)
        
        query = f"""
            SELECT {select_clause} 
            FROM {self.table_name}
            WHERE Platte = ?
        """
        params: list = [plate]

        if slot is not None:
            self._ensure_plate_slot_exists(plate, slot)
            query += " AND Nut = ?"
            params.append(slot)
        
        if data_origin:
            query += " AND DataOrigin = ?"
            params.append(data_origin)
            
        if signals:
            signals = list(signals)
            placeholders = ", ".join(["?"] * len(signals)) ## so we are adding this many placeholders as there is the lengt of signals 
            query += f" AND Signal IN ({placeholders})"
            params.extend(signals)
            
        if wcs_min is not None:
            query += " AND WCS_Y_mm >= ?"
            params.append(wcs_min)
            
        if wcs_max is not None:
            query += " AND WCS_Y_mm <= ?"
            params.append(wcs_max)
        
        if order_by_time and ("Time" in self._valid_cols) and (fields is None or "Time" in validated_fields):
            query += " ORDER BY Time"
            
        if limit is not None:
            limit = min(int(limit), self.max_limit)
            query += " LIMIT ?"
            params.append(limit)
        
        #transform the response into Data Frame 
        df = self.query_df(query, params)
        
        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        
        return df
    
    # ----------------------------
    # Place for your queries 
    # ----------------------------

    def slot_metadata_summary(self, plate: str, *, data_origin: Optional[str] = None) -> pd.DataFrame:
        """Return one metadata row per slot for a plate.

        This mirrors the scalar metadata extracted in `summarize_chatter_cases`
        without loading raw rows into Python.

        Columns:
          - Nut (slot id)
          - Nut_ID (slot id as stored in signal R319; falls back to Nut)
          - X_Position_Nut (signal R321)
          - Werkzeugradius (signal actToolRadius[u1])
          - Drehzahl (signal R330)
          - Werkzeug (signal actToolIdent[u1,1], Value_String)
        """

        self._ensure_plate_exists(plate)

        # NOTE: Use deterministic aggregates (MIN/MAX) to keep stable output.
        # The raw table is long-form (Signal/Value[/Value_String]).
        query = f"""
            SELECT
                Nut,
                COALESCE(MIN(CASE WHEN Signal = 'R319' THEN Value END), Nut) AS Nut_ID,
                MIN(CASE WHEN Signal = 'R321' THEN Value END) AS X_Position_Nut,
                MIN(CASE WHEN Signal = 'actToolRadius[u1]' THEN Value END) AS Werkzeugradius,
                MIN(CASE WHEN Signal = 'R330' THEN Value END) AS Drehzahl,
                MIN(CASE WHEN Signal = 'actToolIdent[u1,1]' THEN Value_String END) AS Werkzeug
            FROM {self.table_name}
            WHERE Platte = ?
              AND Nut IS NOT NULL
              AND Signal IN (
                'R319',
                'R321',
                'R330',
                'actToolRadius[u1]',
                'actToolIdent[u1,1]'
              )
        """
        params: list = [plate]

        if data_origin:
            query += " AND DataOrigin = ?"
            params.append(data_origin)

        query += " GROUP BY Nut ORDER BY Nut"
        return self.query_df(query, params)

    def slot_chatter_cases_summary(self, plate: str, *, data_origin: Optional[str] = None) -> pd.DataFrame:
        """Return chatter boundary rows per slot (long-form).

        Output format matches `summarize_chatter_cases` shape at the boundary:
          - One row with Chatter=0 if R310 has Value=0 in the slot (Y_max is max WCS_Y_mm where Value=0).
          - One row with Chatter=1 if R310 has Value=1 in the slot (Y_max is the slot's max WCS_Y_mm).

        Columns: Nut, Chatter, Y_max
        """

        self._ensure_plate_exists(plate)

        query = f"""
            WITH SlotAgg AS (
                SELECT
                    Nut,
                    MAX(WCS_Y_mm) AS slot_y_max,
                    MAX(CASE WHEN Signal = 'R310' AND Value = 0 THEN WCS_Y_mm END) AS y_max_no_chatter,
                    MAX(CASE WHEN Signal = 'R310' AND Value = 1 THEN 1 ELSE 0 END) AS has_chatter
                FROM {self.table_name}
                WHERE Platte = ?
                  AND Nut IS NOT NULL
                  AND WCS_Y_mm IS NOT NULL
            """
        params: list = [plate]

        if data_origin:
            query += " AND DataOrigin = ?"
            params.append(data_origin)

        query += """
                GROUP BY Nut
            )
            SELECT
                Nut,
                0 AS Chatter,
                y_max_no_chatter AS Y_max
            FROM SlotAgg
            WHERE y_max_no_chatter IS NOT NULL

            UNION ALL

            SELECT
                Nut,
                1 AS Chatter,
                slot_y_max AS Y_max
            FROM SlotAgg
            WHERE has_chatter = 1

            ORDER BY Nut, Chatter
        """

        return self.query_df(query, params)
    
    # Example:
    #
    # def my_query(self, arg1, arg2):
    #     self._ensure_plate_exists(arg1)
    #     query = f"""
    #         SELECT ... FROM {self.table_name}
    #         WHERE Platte = ? AND Nut = ?
    #     """
    #     params = [arg1, arg2]
    #     # (Optional) Add more filters/logic
    #     df = self.query_df(query, params)
    #     return df
    
    # ----------------------------
    # Closing the connection to DB
    # ----------------------------
    
    # function to safely close the connection 
    def close(self) -> None:
        try:
            if getattr(self, "con", None) is not None:
                self.con.close()
        except Exception:
            pass
        finally: 
            self.con = None




        
             
            



