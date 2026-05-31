import os
import sys
import argparse
import questionary
import cleaner_controller

if __name__ == "__main__":
    parent_directory = "./recordings"
    CANCEL_OPTION = "Abbrechen"
    
    if not os.path.exists(parent_directory):
        print(f"Fehler: Der Ordner '{parent_directory}' existiert nicht.")
        sys.exit(1)

    recording_folders = [
        f for f in os.listdir(parent_directory) 
        if os.path.isdir(os.path.join(parent_directory, f))
    ]

    parser = argparse.ArgumentParser(description="Interaktives Menü oder direkte Bereinigung von Aufzeichnungen.")
    parser.add_argument(
        "folder_name", 
        nargs="?", 
        type=str, 
        help="Optional: Der Name des Ordners, der direkt verarbeitet werden soll (überspringt das Menü)."
    )
    
    args = parser.parse_args()

    if args.folder_name:
        selected_folder = args.folder_name
        
        if selected_folder and selected_folder not in recording_folders:
                print(f"Fehler: Der Ordner '{selected_folder}' wurde nicht gefunden.\n")
                selected_folder = None 

                if not selected_folder:
                    if not recording_folders:
                        print(f"Keine Aufzeichnungen im Ordner '{parent_directory}' gefunden.")
                        sys.exit(0)
                        
                    choices = recording_folders + [CANCEL_OPTION]
                        
                    selected_folder = questionary.select(
                        "Welche Aufzeichnung möchtest du bereinigen?",
                        choices=choices,
                        instruction="(Navigiere mit den Pfeiltasten hoch/runter und bestätige mit Enter)"
                    ).ask()
                
                if selected_folder == CANCEL_OPTION or not selected_folder:
                    print("\nVorgang abgebrochen.")
                    sys.exit(0)
            
        full_path = os.path.join(parent_directory, selected_folder)
            
    else:
        if not recording_folders:
            print(f"Keine Aufzeichnungen im Ordner '{parent_directory}' gefunden.")
            sys.exit(0)
            
        choices = recording_folders + [CANCEL_OPTION]
            
        selected_folder = questionary.select(
            "Welche Aufzeichnung möchtest du bereinigen?",
            choices=choices,
            instruction="(Navigiere mit den Pfeiltasten hoch/runter und bestätige mit Enter)"
        ).ask()
        
        if selected_folder == CANCEL_OPTION or not selected_folder:
            print("\nVorgang abgebrochen.")
            sys.exit(0)
            
        full_path = os.path.join(parent_directory, selected_folder)

    print(f"\nStarte Bereinigung für: {full_path}\n" + "-"*40)
    cleaner_controller.clean_all_recordings(full_path)