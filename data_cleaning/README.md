# Projekt-Setup

Diese Anleitung beschreibt die **lokale** Ausführung: welche Daten vorab bereitgestellt werden müssen und wie Skript und Notebooks gestartet werden. Für den Betrieb auf dem Server gibt es eine eigene Anleitung, diese befinded sich auf dem Server in der Arbeitsumgebung.

## Installation

Benötigt Python 3.9 oder neuer.

### Virtuelle Umgebung anlegen

Optional, aber empfohlen, wenn die Abhängigkeiten vom System getrennt bleiben sollen:

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD)**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

> Hinweis: Falls PowerShell die Aktivierung mit einer Fehlermeldung zur Execution Policy blockiert, hilft einmalig `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

### Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### Notebooks in VS Code ausführen

Die beiden Notebooks (`verification.ipynb` und `performance.ipynb`) lassen sich direkt in VS Code öffnen und ausführen:

1. Die Erweiterungen **Python** und **Jupyter** (beide von Microsoft) in VS Code installieren.
2. Den Projektordner in VS Code öffnen und die gewünschte `.ipynb`-Datei anklicken.
3. Oben rechts über **Select Kernel** den Interpreter der virtuellen Umgebung auswählen (`.venv`). Falls `.venv` nicht in der Liste auftaucht: **Select Another Kernel → Python Environments** und dort den Pfad wählen (`.venv/bin/python` bzw. unter Windows `.venv\Scripts\python.exe`).
4. Einzelne Zellen mit dem Play-Symbol oder `Shift + Enter` starten, alle Zellen über **Run All**.

> **Hinweis:** Beim ersten Start fragt VS Code eventuell nach dem Paket `ipykernel`. Die angebotene Installation bestätigen — sie erfolgt in die aktive virtuelle Umgebung.

## Voraussetzungen

Alle benötigten Dateien liegen in der Cloud. Sie sind nicht Teil des Repositories und müssen manuell heruntergeladen und in die unten genannten Ordner eingefügt werden.

In der Cloud sind die Dateien wie folgt abgelegt:

| Inhalt | Speicherort in der Cloud |
| --- | --- |
| Neue Rohdateien | `raw_data/new` |
| Alte Rohdaten | `raw_data/old` |
| Datenbanken | `databases` |

## Ordnerstruktur

```
.
├── data/
│   ├── raw/        # neue Rohdateien
│   └── raw_old/    # alte Rohdaten (eine JSON-Datei)
├── database/       # DBnew.duckdb und DBold.duckdb
├── scripts/
│   └── selector.py  # start des Cleanup-Skriptes (cleaner.py)
├── verification/
│   └── verification.ipynb
└── performance/
    └── performance.ipynb
```

## 1. Cleanup-Skript ausführen

Das Skript liegt in `./scripts/`.

1. Die neuen Rohdateien aus der Cloud unter `raw_data/new` herunterladen.
2. Die Archive entpacken.
3. Den Inhalt in den Ordner `./data/raw/` einfügen.

Anschließend in den Ordner `./scripts/` wechseln und das Skript starten:

```bash
cd scripts
python selector.py
```

Das Skript listet alle Aufzeichnungen aus `../data/raw` auf. Die zuvor eingefügte Aufzeichnung auswählen und mit Enter bestätigen.

Alle Pfade im Skript sind bereits korrekt gesetzt und müssen nicht angepasst werden. Die bereinigten Daten werden von `cleaner_controller.py` in eine DuckDB-Datei geschrieben.

## 2. Verification Notebook ausführen

Das Notebook liegt unter `./verification/verification.ipynb`.

1. Die Datenbanken `DBnew.duckdb` und `DBold.duckdb` aus der Cloud unter `./databases` herunterladen.
2. Beide Dateien in den lokalen Ordner `./database/` einfügen.

> Hinweis: Der Ordner in der Cloud heißt `databases` (Plural), der lokale Zielordner `database` (Singular).

## 3. Performance Notebook ausführen

Das Notebook liegt unter `./performance/performance.ipynb`.

Für dieses Notebook werden sowohl die neuen als auch die alten Rohdaten benötigt:

1. Die neuen Rohdateien aus `raw_data/new` in `./data/raw/` einfügen (siehe Schritt 1).
2. Die alte Rohdatei (eine einzelne JSON-Datei) aus `raw_data/old` in `./data/raw_old/` einfügen.

## Übersicht

| Was soll ausgeführt werden? | Pfad | Benötigte Daten | Quelle in der Cloud | Zielordner |
| --- | --- | --- | --- | --- |
| Cleanup-Skript | `./scripts/selector.py` | Neue Rohdateien (entpackt) | `raw_data/new` | `./data/raw/` |
| Verification Notebook | `./verification/verification.ipynb` | `DBnew.duckdb`, `DBold.duckdb` | `databases` | `./database/` |
| Performance Notebook | `./performance/performance.ipynb` | Neue Rohdateien (entpackt) | `raw_data/new` | `./data/raw/` |
| | | Alte Rohdaten (JSON) | `raw_data/old` | `./data/raw_old/` |