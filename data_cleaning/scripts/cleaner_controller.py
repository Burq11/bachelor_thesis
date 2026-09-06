"""Cleanup-Pipeline.

Gegenueber der Vorversion aendert sich ausschliesslich das Error-Handling:
Fehler werden als PipelineError (oder Subklasse) geworfen statt geprintet
und verschluckt. Jeder Fehler traegt einen status_code mit HTTP-Semantik.
Die Verarbeitungslogik, die Pfadbehandlung (Strings, os.path) und das SQL
sind unveraendert.
"""

from pathlib import Path
import json
import os
import sys
import traceback

import cleaner
import duckdb


def info(msg):
    print(msg)


def error(msg):
    print(msg, file=sys.stderr)


# --------------------------------------------------------------------------
# Fehlertypen
# --------------------------------------------------------------------------

class PipelineError(Exception):
    """Basisklasse. status_code entspricht der HTTP-Semantik."""

    status_code = 500

    def __init__(self, message, source=None):
        super().__init__(message)
        self.source = str(source) if source is not None else None

    def to_dict(self):
        """Serialisierbare Form, z.B. als JSON-Response."""
        return {
            "error": type(self).__name__,
            "detail": str(self),
            "source": self.source,
            "status_code": self.status_code,
        }


class NotFoundError(PipelineError):
    """Erwartete Datei/Ordner existiert nicht oder ist leer."""
    status_code = 404


class InvalidMetadataError(PipelineError):
    """Datei existiert, ist aber inhaltlich kaputt oder unvollstaendig."""
    status_code = 422


class ProcessingError(PipelineError):
    """Fehler waehrend der Verarbeitung einer Aufnahme."""
    status_code = 500


class DatabaseError(PipelineError):
    """Fehler beim Schreiben in DuckDB."""
    status_code = 500


class PartialFailureError(PipelineError):
    """Ein Teil der Aufnahmen ist durchgelaufen, ein Teil nicht."""
    status_code = 207

    def __init__(self, failures, succeeded):
        msg = "%d von %d Aufnahmen fehlgeschlagen." % (
            len(failures), len(failures) + len(succeeded)
        )
        super().__init__(msg)
        self.failures = failures
        self.succeeded = succeeded

    def to_dict(self):
        data = super().to_dict()
        data["succeeded"] = self.succeeded
        data["failures"] = [f.to_dict() for f in self.failures]
        return data


# --------------------------------------------------------------------------

def get_recording_dirs(base_directory):
    if not os.path.exists(base_directory):
        raise NotFoundError(
            "Basisverzeichnis existiert nicht: %s" % base_directory, base_directory
        )
    if not os.path.isdir(base_directory):
        raise NotFoundError(
            "Basispfad ist kein Verzeichnis: %s" % base_directory, base_directory
        )

    try:
        dirs = [os.path.join(base_directory, d) for d in os.listdir(base_directory)
                if os.path.isdir(os.path.join(base_directory, d))]
    except OSError as e:
        raise PipelineError(
            "Basisverzeichnis nicht lesbar: %s" % e, base_directory
        )

    if not dirs:
        raise NotFoundError(
            "Keine Aufnahme-Ordner in %s gefunden." % base_directory, base_directory
        )

    return dirs


def get_nut_platte_from_tags(recording_dir):
    session_dir = os.path.dirname(recording_dir)
    meta_json_path = os.path.join(session_dir, "metadata.json")
    folder_name = os.path.basename(recording_dir)

    if not os.path.isfile(meta_json_path):
        raise NotFoundError(
            "metadata.json fehlt: %s" % meta_json_path, recording_dir
        )

    try:
        with open(meta_json_path, 'r') as f:
            meta = json.load(f)
    except ValueError as e:
        # json.JSONDecodeError ist Subklasse von ValueError
        raise InvalidMetadataError(
            "metadata.json ist kein gueltiges JSON: %s" % e, meta_json_path
        )
    except OSError as e:
        raise PipelineError(
            "metadata.json nicht lesbar: %s" % e, meta_json_path
        )

    if not isinstance(meta, dict):
        raise InvalidMetadataError(
            "metadata.json enthaelt kein Objekt.", meta_json_path
        )

    tags = meta.get("tags", {}).get(folder_name)
    if tags is None:
        raise InvalidMetadataError(
            "Keine Tags fuer '%s' in %s gefunden." % (folder_name, meta_json_path),
            recording_dir,
        )

    nut = tags.get("Nutnummer")
    platte = tags.get("Plattennummer")
    if nut is None or platte is None:
        raise InvalidMetadataError(
            "Plattennummer oder Nutnummer fehlt in '%s' in %s."
            % (folder_name, meta_json_path),
            recording_dir,
        )

    try:
        return int(nut), int(platte)
    except (TypeError, ValueError):
        raise InvalidMetadataError(
            "Nutnummer/Plattennummer sind nicht numerisch in '%s'." % folder_name,
            recording_dir,
        )


def clean_one_recording(recording_dir):
    nut, platte = get_nut_platte_from_tags(recording_dir)

    try:
        output_path = cleaner.cleaner(recording_dir, nut=nut, platte=platte)
    except PipelineError:
        raise
    except Exception as e:
        raise ProcessingError(
            "cleaner() ist fehlgeschlagen: %s: %s" % (type(e).__name__, e),
            recording_dir,
        )

    return output_path, recording_dir


def build_database(base_directory):
    p = Path("../data/interim")

    if not p.is_dir():
        raise NotFoundError("Interim-Verzeichnis existiert nicht: %s" % p, p)

    files_to_process = list(p.iterdir())
    if not files_to_process:
        raise NotFoundError(
            "Keine Parquet-Dateien zum Laden in die Datenbank gefunden.", p
        )

    try:
        con = duckdb.connect('../data/processed/duckdb.duckdb')
    except duckdb.Error as e:
        raise DatabaseError("Verbindung zur DuckDB fehlgeschlagen: %s" % e)

    try:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS cleaned_data AS 
            SELECT * FROM read_parquet('{p}') LIMIT 0
        """)

        con.execute(f"""
            CREATE TEMP VIEW incoming_data AS 
            SELECT * FROM read_parquet('{p}')
        """)

        overlap_count = con.execute("""
            DELETE FROM cleaned_data 
            WHERE EXISTS (
                SELECT 1 
                FROM incoming_data 
                WHERE incoming_data.Nut = cleaned_data.Nut 
                  AND incoming_data.Platte = cleaned_data.Platte
            )
        """).fetchone()[0]

        if overlap_count > 0:
            info("%d existierende Eintraege gefunden. Werden ueberschrieben..." % overlap_count)
        else:
            info("Keine Ueberschneidungen gefunden. Fuege komplett neue Daten hinzu...")

        con.execute("""
            INSERT INTO cleaned_data 
            SELECT * FROM incoming_data
            ORDER BY Platte, Nut, Signal, Time
        """)

        con.execute("DROP VIEW incoming_data")

    except duckdb.Error as e:
        raise DatabaseError("Laden in die Datenbank fehlgeschlagen: %s" % e)
    finally:
        con.close()

    info("Data loaded successfully. Deleting source files...")

    deleted_count = 0
    for file_path in files_to_process:
        try:
            os.remove(file_path)
            deleted_count += 1
        except OSError as e:
            # Kein harter Fehler: die Daten liegen bereits in der DB.
            error("Fehler beim Loeschen der interim Datei %s: %s" % (file_path, e))

    info("Successfully deleted %d parquet files." % deleted_count)
    return deleted_count


def clean_all_recordings(base_directory, fail_fast=False):
    recording_dirs = get_recording_dirs(base_directory)   # wirft 404, wenn leer
    total = len(recording_dirs)
    info("Starte sequenzielle Verarbeitung fuer %d Aufzeichnungen..." % total)

    succeeded = []
    failures = []

    for i, recording_dir in enumerate(recording_dirs):
        folder_name = os.path.basename(recording_dir)
        try:
            output_path, rec_dir = clean_one_recording(recording_dir)
        except PipelineError as e:
            error("[%d/%d] %s: %s" % (i + 1, total, folder_name, e))
            # 500er sind unerwartet -> Original-Traceback zeigen.
            # 404/422 sind Datenprobleme, die Meldung reicht.
            cause = e.__cause__ or e.__context__
            if e.status_code >= 500 and cause is not None:
                traceback.print_exception(type(cause), cause, cause.__traceback__)
            if fail_fast:
                raise
            failures.append(e)
        except Exception as e:
            wrapped = ProcessingError(
                "Unerwarteter Fehler: %s: %s" % (type(e).__name__, e), recording_dir
            )
            error("[%d/%d] %s: %s" % (i + 1, total, folder_name, wrapped))
            traceback.print_exc()
            if fail_fast:
                raise wrapped
            failures.append(wrapped)
        else:
            info("[%d/%d] %s OK" % (i + 1, total, folder_name))
            succeeded.append(folder_name)

    if not succeeded:
        # Nichts durchgekommen: den ersten Fehler durchreichen.
        raise failures[0]

    deleted = build_database(base_directory)

    if failures:
        raise PartialFailureError(failures, succeeded)

    return {"status_code": 200, "processed": len(succeeded), "files_loaded": deleted}


EXIT_CODES = {404: 2, 422: 3, 207: 4, 500: 1}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if not argv:
        error("Usage: cleaner_controller.py <base_directory>")
        return 64

    try:
        result = clean_all_recordings(argv[0])
    except KeyboardInterrupt:
        error("Abgebrochen.")
        return 130
    except PipelineError as e:
        error("[%d] %s" % (e.status_code, json.dumps(e.to_dict(), ensure_ascii=False)))
        return EXIT_CODES.get(e.status_code, 1)

    info("Fertig: %d Aufnahmen, %d Dateien geladen."
         % (result["processed"], result["files_loaded"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())