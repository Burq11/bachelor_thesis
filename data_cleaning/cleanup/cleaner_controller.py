import cleaner
import duckdb
import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def get_recording_dirs(base_directory):
    return [os.path.join(base_directory, d) for d in os.listdir(base_directory) 
            if os.path.isdir(os.path.join(base_directory, d))]

def clean_one_recording(args):
    recording_dir, r_param = args
    worker_name = multiprocessing.current_process().name
    
    output_path = cleaner.cleaner(recording_dir, r_parameter=r_param)
    return output_path, worker_name, recording_dir

def build_database(base_directory):
    con = duckdb.connect('./duckdb.duckdb')
    try:
        con.execute("DROP TABLE IF EXISTS cleaned_data")
        
        glob_pattern = f"{base_directory}/**/cleaned_data.parquet"
        
        con.execute(f"""
            CREATE TABLE cleaned_data AS 
            SELECT * FROM read_parquet('{glob_pattern}')
        """)
        
        con.table("cleaned_data").limit(5).show()
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
            
            output_path, worker_name, rec_dir = result
            folder_name = os.path.basename(rec_dir)
            
            if output_path:
                print(f"[{i+1}/{len(tasks)}] ✅ {folder_name} (Parquet erstellt)")
            else:
                print(f"[{i+1}/{len(tasks)}] ⚠️ {folder_name} (Übersprungen/Fehler)")
                
        except Exception as e:
            folder_name = os.path.basename(task[0])
            print(f"Fehler bei {folder_name}: {e}")
    build_database(base_directory)


import glob
import os

def cleanup_parquet_files(base_directory, dry_run=True):
    """
    Löscht NUR die erstellten 'cleaned_data.parquet' Dateien.
    Mit dry_run=True wird nur angezeigt, was passieren würde.
    """
    if dry_run:
        print("\nStarte Aufräumaktion DRY RUN...")
    else:
        print("\nStarte Aufräumaktion...")
    
    glob_pattern = os.path.join(base_directory, "**", "cleaned_data.parquet")
    dateien = glob.glob(glob_pattern, recursive=True)
    
    geloescht = 0
    for datei in dateien:
        if os.path.basename(datei) != "cleaned_data.parquet":
            print(f"Datei {datei} weicht vom Namen ab. Wird übersprungen!")
            continue

        if dry_run:
            print(f"Würde löschen: {datei}")
        else:
            try:
                os.remove(datei)
                geloescht += 1
            except OSError as e:
                print(f"⚠️ Fehler beim Löschen von {datei}: {e}")
            
    if dry_run:
        print(f"Es würden {len(dateien)} bereinigte Dateien gelöscht werden.")
    else:
        print(f"{geloescht} Parquet-Dateien erfolgreich gelöscht!")