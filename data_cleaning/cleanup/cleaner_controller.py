from pathlib import Path
import cleaner
import duckdb
import json
import os
import glob

def get_recording_dirs(base_directory):
    return [os.path.join(base_directory, d) for d in os.listdir(base_directory)
            if os.path.isdir(os.path.join(base_directory, d))]

def get_nut_platte_from_tags(recording_dir):
    session_dir = os.path.dirname(recording_dir)
    meta_json_path = os.path.join(session_dir, "metadata.json")
    folder_name = os.path.basename(recording_dir)

    with open(meta_json_path, 'r') as f:
        meta = json.load(f)

    tags = meta.get("tags", {}).get(folder_name)
    if tags is None:
        raise ValueError(f"Keine Tags für '{folder_name}' in {meta_json_path} gefunden.")

    nut = tags.get("R-Parameter Tag (R300)")
    platte = tags.get("R-Parameter Tag (R301)")
    if nut is None or platte is None:
        raise ValueError(f"R300/R301-Tag fehlt für '{folder_name}' in {meta_json_path}.")

    return int(nut), int(platte)

def clean_one_recording(recording_dir):
    nut, platte = get_nut_platte_from_tags(recording_dir)
    output_path = cleaner.cleaner(recording_dir, nut=nut, platte=platte)
    return output_path, recording_dir

def build_database(base_directory):
    con = duckdb.connect('./duckdb.duckdb')
    try:
        glob_pattern = Path(base_directory).joinpath("**", "cleaned_data.parquet").as_posix()        
        files_to_process = glob.glob(glob_pattern, recursive=True)
        if not files_to_process:
            print("Keine Parquet-Dateien zum Laden in die Datenbank gefunden.")
            return

        con.execute(f"""
            CREATE TABLE IF NOT EXISTS cleaned_data AS 
            SELECT * FROM read_parquet('{glob_pattern}') LIMIT 0
        """)
        
        con.execute(f"""
            CREATE TEMP VIEW incoming_data AS 
            SELECT * FROM read_parquet('{glob_pattern}')
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
            print(f"{overlap_count} existierende Einträge gefunden. Werden überschrieben...")
        else:
            print("🆕 Keine Überschneidungen gefunden. Füge komplett neue Daten hinzu...")
        
        con.execute("""
            INSERT INTO cleaned_data 
            SELECT * FROM incoming_data
        """)
        
        con.execute("DROP VIEW incoming_data")
        
        print("Data loaded successfully. Deleting source files...")
        
        deleted_count = 0
        for file_path in files_to_process:
            try:
                os.remove(file_path)
                deleted_count += 1
            except OSError as e:
                print(f"Fehler beim Löschen von {file_path}: {e}")
                
        print(f"Successfully deleted {deleted_count} parquet files.")

    except Exception as e:
        print(f"Fehler beim Erstellen der Datenbank: {e}")
    finally:
        con.close()

def clean_all_recordings(base_directory):
    recording_dirs = get_recording_dirs(base_directory)
    if not recording_dirs:
        print("Keine Ordner gefunden.")
        return

    print(f"Starte sequenzielle Verarbeitung für {len(recording_dirs)} Aufzeichnungen...")

    for i, recording_dir in enumerate(recording_dirs):
        print(recording_dir)
        try:
            result = clean_one_recording(recording_dir)

            output_path, rec_dir = result
            folder_name = os.path.basename(rec_dir)
            
            if output_path:
                print(f"[{i+1}/{len(recording_dirs)}] ✅ {folder_name} (Parquet erstellt)")
            else:
                print(f"[{i+1}/{len(recording_dirs)}] ⚠️ {folder_name} (Übersprungen/Fehler)")

        except Exception as e:
            folder_name = os.path.basename(recording_dir)
            print(f"Fehler bei {folder_name}: {e}")
    build_database(base_directory)