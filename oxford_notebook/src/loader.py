
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

class DataNotFoundError(ValueError):
    """Raised when requested Plate/Nut does not exist, or a filter matches no rows."""
    pass

class InvalidColumnError(ValueError):
    """Raised when requested columns are not part of the table schema"""
    pass

class QueryValidationError(ValueError):
    """
    Raised when query inputs reference columns or filter values that don't exist.
    """
    def __init__(self, issues: list[tuple[str, list[str]]]):
        self.issues = issues
        super().__init__("; ".join(msg for msg, _ in issues))

class NoSignalBearingChannelsError(DataNotFoundError):
    """Origin has rows but none carry a WCS_Y_mm position (e.g. disconnected-sensor
    noise), so there is no plottable vibration signal. Raised by
    `get_axiswise_plot_df`; revisit when real accelerometer data lands."""
    def __init__(self, plate, slot, data_origin: str):
        self.plate, self.slot, self.data_origin = plate, slot, data_origin
        super().__init__(f"DataOrigin={data_origin!r} has no WCS_Y_mm position for "
                         f"plate={plate}, slot={slot}: nothing to plot.")

class DuckDBLoader:
    def __init__(
        self,
        db_path: Path,
        table_name: Optional[str] = None,
        read_only: bool = False,
    ):
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.con = duckdb.connect(str(self.db_path), read_only= read_only)
        # table_name=None means "auto-detect the single table in this database"
        self.table_name = table_name if table_name is not None else self._detect_table()
        self._valid_cols = self._load_schema_cols()

    def _detect_table(self) -> str:
        tables = self.con.execute("SHOW TABLES").fetchall()
        if not tables:
            raise DataNotFoundError(f"No tables found in DuckDB database: {self.db_path}")
        return tables[0][0]
      
        
    # ----------------------------
    # Utilities
    # ----------------------------
    def schema(self) -> pd.DataFrame:
        rows = self.con.execute(f"DESCRIBE {self.table_name}").fetchdf()
        return rows 
    
    def _load_schema_cols(self) -> set[str]:
        rows = self.con.execute(f"DESCRIBE {self.table_name}").fetchall()
        return {row[0] for row in rows} 
    
    # ----------------------------
    # Public query interface
    # ----------------------------
    # These methods are preapared for other modules
    # (e.g. data_processing) to execute SQL. They keep the DuckDB
    # connection encapsulated in the loader: callers author SQL,
    # the loader executes it.

    def query_df(self, sql: str, params: Iterable = ()) -> pd.DataFrame:
        """Execute a parameterised query and return the result as a DataFrame."""
        return self.con.execute(sql, list(params)).df()

    def query_row(self, sql: str, params: Iterable = ()) -> tuple | None:
        """Execute a parameterised query and return the first row, or None."""
        return self.con.execute(sql, list(params)).fetchone()
     
    # Fetches data using get_data_df and applies groupby/aggregation.
    def get_grouped_data(self, group_by: list[str], agg: dict, *args, **kwargs) -> pd.DataFrame:
        kwargs.setdefault("fields", list(dict.fromkeys(list(group_by) + list(agg))))
        df = self.get_data_df(*args, **kwargs)
        return df.groupby(group_by).agg(agg).reset_index() 
    
    # ----------------------------
    # Basic SQL queries
    # ----------------------------
    def list_plates(self) -> list[int]:
        query = f"""
            SELECT DISTINCT Platte
            FROM {self.table_name}
            WHERE Platte IS NOT NULL
            ORDER BY Platte
        """
        result_rows = self.con.execute(query).fetchall()
        return [row[0] for row in result_rows]
    
    
    def list_slots(self) -> list[int]:
        query = f"""
            SELECT DISTINCT Nut
            FROM {self.table_name}
            WHERE Nut IS NOT NULL
            ORDER BY Nut
        """
        result_rows = self.con.execute(query).fetchall()
        return [row[0] for row in result_rows]
    
    
    ## maybe we dont need this, it just shows all the slots of plates
    def list_plate_slots_flat(self) -> list[tuple[int, int]]:
        query = f"""
            SELECT DISTINCT Platte, Nut
            FROM {self.table_name}
            WHERE Platte IS NOT NULL
            AND Nut IS NOT NULL
            ORDER BY Platte, Nut
        """
        result_rows = self.con.execute(query).fetchall()
        return [(row[0], row[1]) for row in result_rows]
    
    
    def list_slots_for_plate(self, plate: int) -> list[int]:
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
    
    
    def list_signals(self, plate: int, slot: Optional[int] = None, data_origin: Optional[str] = None) -> list[str]: 
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

        return df["Signal"].tolist()

    def list_data_origins(self, plate: int, slot: Optional[int] = None) -> list[str]:
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
            
    
    def get_data_df(self, plate: int, slot: Optional[int] = None, *,
        fields: Optional[Iterable[str]] = None, data_origin: Optional[str] = None,
        signals: Optional[Iterable[str]] = None, wcs_min: Optional[float] = None, 
        wcs_max: Optional[float] = None, order_by_time: bool = True, limit: Optional[int] = None) -> pd.DataFrame:

        validated_fields = self._validate_fields(fields)
        select_clause = "*" if validated_fields is None else ", ".join(validated_fields)

        specs = self._filter_specs(plate, slot, data_origin=data_origin, signals=signals,
                                   wcs_min=wcs_min, wcs_max=wcs_max)
        where_sql, params = self._where_sql(specs)

        query = f"""
            SELECT {select_clause}
            FROM {self.table_name}
            WHERE {where_sql}
        """

        if order_by_time and ("Time" in self._valid_cols) and (validated_fields is None or "Time" in validated_fields):
            query += " ORDER BY Time"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        #transform the response into Data Frame
        df = self.query_df(query, params)

        # nothing came back: explain which filter is responsible instead of returning a bare empty frame
        if df.empty:
            raise QueryValidationError(self._diagnose_empty(specs))

        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

        return df

    # ----------------------------
    # Place for your queries
    # ----------------------------

    def slot_metadata_summary(self, plate: int, *, data_origin: Optional[str] = None) -> pd.DataFrame:
        """Return one metadata row per slot for a plate.

        This mirrors the scalar metadata extracted in `summarize_chatter_cases`
        without loading raw rows into Python.

        Columns:
          - Nut (slot id)
          - Nut_ID (slot id as stored in signal R319; falls back to Nut)
          - X_Position_Nut (signal R321)
          - Werkzeugradius (signal actToolRadius[u1])
          - Drehzahl (signal R330)
          - Werkzeug (signal actToolIdent[u1], Value_String)
        """

        metadata_signals = ["R319", "R321", "R330", "actToolRadius[u1]", "actToolIdent[u1,1]"]
        placeholders = ", ".join(["?"] * len(metadata_signals))

        # The signal list and "Nut IS NOT NULL" are structural, not user filters, but they
        # belong in the specs so the diagnosis reports honest numbers if nothing comes back.
        specs = self._filter_specs(plate, data_origin=data_origin, extra=[
            ("slots present", None, "Nut IS NOT NULL", []),
            (f"metadata signals {metadata_signals!r}", None,
             f"Signal IN ({placeholders})", metadata_signals),
        ])
        where_sql, params = self._where_sql(specs)

        # NOTE: Use deterministic aggregates (MIN/MAX) to keep stable output.
        # The raw table is long-form (Signal/Value[/Value_String]).
        query = f"""
            SELECT
                Nut,
                COALESCE(MIN(CASE WHEN Signal = 'R319' THEN Value END), Nut) AS Nut_ID,
                MIN(CASE WHEN Signal = 'R321' THEN Value END) AS X_Position_Nut,
                MIN(CASE WHEN Signal = 'actToolRadius[u1]' THEN Value END) AS Werkzeugradius,
                MIN(CASE WHEN Signal = 'R330' THEN Value END) AS Drehzahl,
                MIN(CASE WHEN Signal = 'actToolIdent[u1]' THEN Value_String END) AS Werkzeug
            FROM {self.table_name}
            WHERE Platte = ?
              AND Nut IS NOT NULL
              AND Signal IN (
                'R319',
                'R321',
                'R330',
                'actToolRadius[u1]',
                'actToolIdent[u1]'
              )
        """

        df = self.query_df(query, params)
        if df.empty:
            raise QueryValidationError(self._diagnose_empty(specs, aggregated=True))
        return df

    def slot_chatter_cases_summary(self, plate: int, *, data_origin: Optional[str] = None) -> pd.DataFrame:
        """Return chatter boundary rows per slot (long-form).

        Output format matches `summarize_chatter_cases` shape at the boundary:
          - One row with Chatter=0 if R310 has Value=0 in the slot (Y_max is max WCS_Y_mm where Value=0).
          - One row with Chatter=1 if R310 has Value=1 in the slot (Y_max is the slot's max WCS_Y_mm).

        Columns: Nut, Chatter, Y_max
        """

        specs = self._filter_specs(plate, data_origin=data_origin)
        where_sql, params = self._where_sql(specs)

        query = f"""
            WITH SlotAgg AS (
                SELECT
                    Nut,
                    MAX(WCS_Y_mm) AS slot_y_max,
                    MAX(CASE WHEN Signal = 'R310' AND Value = 0 THEN WCS_Y_mm END) AS y_max_no_chatter,
                    MAX(CASE WHEN Signal = 'R310' AND Value = 1 THEN 1 ELSE 0 END) AS has_chatter
                FROM {self.table_name}
                WHERE {where_sql}
                  AND Nut IS NOT NULL
                  AND WCS_Y_mm IS NOT NULL
            """
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

        df = self.query_df(query, params)
        if df.empty:
            raise QueryValidationError(self._diagnose_empty(specs, aggregated=True))
        return df
    
    def get_axiswise_plot_df(
        self,
        plate: int,
        slot: Optional[int] = None,
        *,
        data_origin: Optional[str] = None,
        signals: Optional[Iterable[str]] = None,
        wcs_min: Optional[float] = None,
        wcs_max: Optional[float] = None,
        order_by: str = "Time",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Lightweight query for axiswise plotting (External/HF/LF in one DataFrame).

        Returns only columns needed by create_axiswise_plots2 + optional hover fields.
        """

        required_cols = [
            "Platte",
            "Nut",
            "Time",
            "Duration_Seconds",
            "WCS_Y_mm",
            "Axis",
            "Signal",
            "Value",
            "DataOrigin",
            "Groupname",
            "Unit",
            "Description",
            "Label",
            "NcCode",
            "NcComment",
            "IpoGC",
            "Cycle",
        ]
        # keep only columns that actually exist in schema
        fields = [column for column in required_cols if column in self._valid_cols]
        if not fields:
            raise InvalidColumnError("No plot columns found in schema.")

        select_parts = list(fields)
        # Labels for plot annotations
        if "NcCode" in self._valid_cols:
            if "NcComment" in self._valid_cols:
                select_parts.append(
                    "CASE WHEN NcComment IS NOT NULL AND NcComment <> '' "
                    "THEN NcCode || ' ; ' || NcComment ELSE NcCode END AS GCode_Label"
                )
            else:
                select_parts.append("NcCode AS GCode_Label")

        # Labels and Descriptions for signals
        name_sources = [c for c in ("Label", "Description") if c in self._valid_cols]
        if name_sources and "Signal" in self._valid_cols:
            coalesced = ", ".join(name_sources + ["Signal"])
            select_parts.append(f"COALESCE({coalesced}) AS Signal_Label")

        select_clause = ", ".join(select_parts)

        specs = self._filter_specs(
            plate, slot, data_origin=data_origin, signals=signals,
            wcs_min=wcs_min, wcs_max=wcs_max,
            extra=[("Value IS NOT NULL", None, "Value IS NOT NULL", [])])
        where_sql, params = self._where_sql(specs)

        query = f"""
            SELECT {select_clause}
            FROM {self.table_name}
            WHERE {where_sql}
        """

        if order_by in self._valid_cols:
            query += f" ORDER BY {order_by}"
        elif "Time" in self._valid_cols:
            query += " ORDER BY Time"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        df = self.query_df(query, params)

        # nothing came back: explain which filter is responsible instead of returning a bare empty frame
        if df.empty:
            raise QueryValidationError(self._diagnose_empty(specs))

        # rows came back but none carry a position: unplottable, so fail loudly instead
        # of handing callers a frame that renders as empty axes. Keys on WCS_Y_mm only --
        # LF_Data has position but no Axis and must still pass.
        if "WCS_Y_mm" in df.columns and df["WCS_Y_mm"].isna().all():
            raise NoSignalBearingChannelsError(plate, slot, data_origin)

        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

        return df

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
    # Internals: input checks and empty-result diagnosis
    # ----------------------------
    # These build the structured issues carried by QueryValidationError.
    # provider._client_call catches that error and renders the issues for
    # the user, so this machinery is what makes a failed query explain itself.
    # You rarely need to touch this to add a query.

    # ----------------------------
    # Validation
    # ----------------------------
    def _exists(self, where_sql: str, params: list ) -> bool:
        q = f"SELECT 1 FROM {self.table_name} WHERE {where_sql} LIMIT 1"
        return self.con.execute(q,params).fetchone() is not None

    def _ensure_plate_exists(self, plate: int) -> None:
        if not self._exists("Platte = ? AND Platte IS NOT NULL", [plate]):
            raise DataNotFoundError(f"Unknown plate={plate!r}.")
        
    def _ensure_plate_slot_exists(self, plate: int, slot: int) -> None:
            if not self._exists("Platte = ? AND Nut = ?", [plate, slot]):
                if not self._exists("Platte = ?", [plate]):
                    raise DataNotFoundError(f"Unknown plate={plate!r}.")
                raise DataNotFoundError(
                    f"No data for (plate={plate!r}, slot={slot}).")
                
    def _validate_fields(self, fields: Optional[Iterable[str]]) -> Optional[list[str]]:
        """Check requested columns against the schema. Local only, no database roundtrip."""
        if fields is None:
            return None
        fields = list(fields)
        if not fields:
            raise InvalidColumnError(
                "An empty column list is not valid; use fields=None to select all columns.")
        unknown = [c for c in fields if c not in self._valid_cols]
        if unknown:
            raise QueryValidationError([(f"Unknown column(s): {unknown}",
                                         ["Use provider.schema() to see valid columns"])])
        return fields

    # ----------------------------
    # Filter specs and empty-result diagnosis
    # ----------------------------

    def _filter_specs(self, plate: int, slot: Optional[int] = None, *,
        data_origin: Optional[str] = None, signals: Optional[Iterable[str]] = None,
        wcs_min: Optional[float] = None, wcs_max: Optional[float] = None,
        extra: Optional[list[tuple]] = None) -> list[tuple]:
        specs: list[tuple] = [(f"Platte={plate!r}", "Platte", "Platte = ?", [plate])]

        if slot is not None:
            specs.append((f"Nut={slot!r}", "Nut", "Nut = ?", [slot]))

        if data_origin:
            specs.append((f"DataOrigin={data_origin!r}", "DataOrigin", "DataOrigin = ?", [data_origin]))

        if signals:
            signals = list(signals)
            placeholders = ", ".join(["?"] * len(signals))
            specs.append((f"Signal in {signals!r}", "Signal", f"Signal IN ({placeholders})", signals))

        # range filters: listing distinct values would not help, so column is None
        if wcs_min is not None:
            specs.append((f"WCS_Y_mm >= {wcs_min}", None, "WCS_Y_mm >= ?", [wcs_min]))

        if wcs_max is not None:
            specs.append((f"WCS_Y_mm <= {wcs_max}", None, "WCS_Y_mm <= ?", [wcs_max]))

        if extra:
            specs.extend(extra)

        return specs

    @staticmethod
    def _where_sql(specs: list[tuple], skip: Optional[int] = None) -> tuple[str, list]:
        """Join filter specs into a WHERE body, optionally leaving one out."""
        parts: list[str] = []
        params: list = []
        for index, (_, _, sql, spec_params) in enumerate(specs):
            if index == skip:
                continue
            parts.append(sql)
            params.extend(spec_params)
        return (" AND ".join(parts) if parts else "TRUE"), params

    def _suggest_values(self, specs: list[tuple], index: int, limit: int = 20) -> list:
        """Values available for one filter's column once the other filters are applied."""
        column = specs[index][1]
        if column is None:
            return []
        where_sql, params = self._where_sql(specs, skip=index)
        query = f"""
            SELECT DISTINCT {column}
            FROM {self.table_name}
            WHERE {where_sql}
            AND {column} IS NOT NULL
            LIMIT {int(limit)}
        """
        return [row[0] for row in self.con.execute(query, params).fetchall()]

    def _diagnose_empty(self, specs: list[tuple], *,
        aggregated: bool = False) -> list[tuple[str, list[str]]]:
        """
        Explain an empty result by leave-one-out attribution.

        Set aggregated=True for queries that GROUP BY, so that a result which is
        empty despite matching rows is explained by the aggregation rather than
        by a filter. Raw row queries must leave it False.
        """
        described = ", ".join(label for label, _, _, _ in specs)
        if not specs:
            return [("The query returned no rows.", [])]

        # If the filters together do match rows, the emptiness came from something downstream
        where_sql, where_params = self._where_sql(specs)
        matched = self.con.execute(
            f"SELECT COUNT(*) FROM {self.table_name} WHERE {where_sql}", where_params).fetchone()[0]
        if matched:
            if aggregated:
                hints = [f"The filters match {matched:,} rows, so the query returned nothing "
                         "after grouping/aggregation rather than because of a filter.",
                         "The signals this query aggregates are probably absent from that selection."]
            else:
                hints = [f"The filters match {matched:,} rows, so no filter is responsible.",
                         "Check the limit argument: limit=0 (or negative) returns no rows."]
            return [(f"No rows in the result for: {described}", hints)]

        selects: list[str] = []
        params: list = []
        for index in range(len(specs)):
            where_sql, where_params = self._where_sql(specs, skip=index)
            selects.append(f"bool_or({where_sql}) AS c{index}")
            params.extend(where_params)

        query = f"SELECT {', '.join(selects)} FROM {self.table_name}"
        matches = self.con.execute(query, params).fetchone()

        culprits = [index for index, matched in enumerate(matches) if matched]

        # Removing any single filter still returns nothing: not a filter mistake.
        if not culprits:
            return [(f"No rows matched: {described}",
                     ["Removing any single filter still returns nothing, so this selection "
                      "is genuinely empty rather than mis-filtered.",
                      "Use provider.plate_slots() to check the plate and slot exist at all."])]

        hints: list[str] = []
        for index in culprits:
            label, column = specs[index][0], specs[index][1]
            values = self._suggest_values(specs, index)
            if values:
                hints.append(f"Available {column}: {', '.join(str(v) for v in values)}")
            else:
                # range filters have no values to list, so just name the culprit
                hints.append(f"Dropping {label} brings rows back.")

        if len(culprits) > 1:
            hints.append("Each of these filters matches on its own, but they never co-occur.")

        return [(f"No rows matched: {described}", hints)]

    
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




        
             
            



