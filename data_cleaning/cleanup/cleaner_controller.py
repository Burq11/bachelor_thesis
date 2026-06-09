from pathlib import Path
import cleaner
import duckdb
import os
import glob

def get_recording_dirs(base_directory):
    return [os.path.join(base_directory, d) for d in os.listdir(base_directory) 
            if os.path.isdir(os.path.join(base_directory, d))]

def clean_one_recording(args):
    recording_dir, r_param = args
    
    output_path = cleaner.cleaner(recording_dir, r_parameter=r_param)
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
    
    tasks = [(folder, i+1) for i, folder in enumerate(recording_dirs)]
    
    print(f"Starte sequenzielle Verarbeitung für {len(tasks)} Aufzeichnungen...")
    
    for i, task in enumerate(tasks):
        try:
            result = clean_one_recording(task)
            
            output_path, rec_dir = result
            folder_name = os.path.basename(rec_dir)
            
            if output_path:
                print(f"[{i+1}/{len(tasks)}] ✅ {folder_name} (Parquet erstellt)")
            else:
                print(f"[{i+1}/{len(tasks)}] ⚠️ {folder_name} (Übersprungen/Fehler)")
                
        except Exception as e:
            folder_name = os.path.basename(task[0])
            print(f"Fehler bei {folder_name}: {e}")
    build_database(base_directory)