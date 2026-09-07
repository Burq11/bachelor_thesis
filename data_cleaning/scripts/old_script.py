import json
from pathlib import Path

import numpy as np
import pandas as pd


def import_data(dateipfad: str) -> dict:
    """
    Lädt Daten aus einer JSON-Datei und gibt sie als Dictionary zurück.

    Parameters
    ----------
    dateipfad : str
        Der Pfad zur JSON-Datei.

    Returns
    -------
    dict
        Die geladenen Daten aus der JSON-Datei.

    Raises
    ------
    FileNotFoundError
        Wenn die Datei nicht gefunden wird.
    json.JSONDecodeError
        Wenn die Datei kein gültiges JSON enthält.
    """
    try:
        # Lade die gesamte JSON-Datei
        with open(dateipfad, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Die Datei {dateipfad} wurde nicht gefunden.")
        raise
    except json.JSONDecodeError:
        print(f"Die Datei {dateipfad} enthält kein gültiges JSON.")
        raise
    return data


def import_meta_df(data):
    """
    Importiert Meta-Daten aus einem gegebenen Dictionary und formatiert sie in ein DataFrame.

    Parameters
    ----------
    data : dict
        Das Eingabedatensatz im Dictionary-Format, das die Schlüssel 'Header' und 'Footer' enthält.

    Returns
    -------
    pd.DataFrame
        Ein DataFrame mit den Meta-Daten aus 'Header' und 'Footer'.
    """
    header_footer_records = []

    # Header verarbeiten
    for k, v in data["Header"].items():
        if isinstance(v, (list, dict)):
            if isinstance(v, dict):
                if k == "Initial":
                    for l, kl_value in v.items():
                        header_footer_records.append(
                            {"key": f"{k}_{l}", "value": kl_value, "origin": "header"}
                        )
                else:
                    for l, kl_value in v.items():
                        header_footer_records.append(
                            {"key": l, "value": kl_value, "origin": "header"}
                        )
            elif isinstance(v, list):
                if k == "JobDescription":
                    for item in v:
                        if isinstance(item, str):
                            item = json.loads("{" + item + "}")
                            if isinstance(item, dict):
                                for l, kl_value in item.items():
                                    header_footer_records.append(
                                        {
                                            "key": f"{k}_{l}",
                                            "value": kl_value,
                                            "origin": "header",
                                        }
                                    )
        else:
            header_footer_records.append({"key": k, "value": v, "origin": "header"})

    # Footer verarbeiten
    for k, v in data["Footer"].items():
        if isinstance(v, dict):
            if k == "FilePathChain":
                for l, kl_value in v.items():
                    header_footer_records.append(
                        {"key": f"{k}_{l}", "value": kl_value, "origin": "footer"}
                    )
            else:
                for l, kl_value in v.items():
                    header_footer_records.append(
                        {"key": l, "value": kl_value, "origin": "footer"}
                    )

    return pd.DataFrame(header_footer_records)


def import_signalListHF(data):
    """
    Importiert die HF-Signalliste aus dem gegebenen Datensatz und formatiert sie in ein DataFrame.

    Parameters
    ----------
    data : dict
        Das Eingabedatensatz im Dictionary-Format, das den Schlüssel 'Header' enthält.

    Returns
    -------
    pd.DataFrame
        Ein DataFrame mit den HF-Signaldaten, bei dem die Spalten 'Name' und 'Address' umbenannt bzw. entfernt wurden.
    """
    signal_list_hf_records = []

    # Überprüfen, ob 'SignalListHFData' im Header vorhanden ist
    if "SignalListHFData" in data["Header"]:
        for item in data["Header"]["SignalListHFData"]:
            signal_list_hf_records.append(item)

    # DataFrame erstellen
    df_signalListHF = pd.DataFrame(signal_list_hf_records)

    # Überprüfen, ob DataFrame nicht leer ist, bevor Spalten umbenannt/entfernt werden
    if not df_signalListHF.empty:
        df_signalListHF.rename(columns={"Name": "Signal"}, inplace=True)
        df_signalListHF.drop(columns=["Address"], inplace=True, errors="ignore")

    return df_signalListHF


def import_signalListLF(data):
    """
    Importiert die LF-Signalliste aus dem gegebenen Datensatz und formatiert sie in ein DataFrame.

    Parameters
    ----------
    data : dict
        Das Eingabedatensatz im Dictionary-Format, das den Schlüssel 'Header' enthält.

    Returns
    -------
    pd.DataFrame
        Ein DataFrame mit den LF-Signaldaten, wobei 'Category' und 'Signal' aus der 'path'-Spalte extrahiert wurden.
    """
    signal_list_lf_records = []

    # Überprüfen, ob 'SignalListLFData' im Header vorhanden ist
    if "SignalListLFData" in data["Header"]:
        for item in data["Header"]["SignalListLFData"]:
            signal_list_lf_records.append(item)

    # DataFrame erstellen
    df_signalListLF = pd.DataFrame(signal_list_lf_records)

    # Überprüfen, ob DataFrame nicht leer ist, bevor Spalten bearbeitet werden
    if not df_signalListLF.empty:
        if "path" in df_signalListLF.columns:
            df_signalListLF["Category"] = df_signalListLF.path.str.split("/").str[2]
            df_signalListLF["Signal"] = df_signalListLF.path.str.split("/").str[3]

            # 🔧 RXXX-Umbenennung für r[u1,XXX,#1]-Signale
            df_signalListLF["Signal"] = df_signalListLF["Signal"].str.replace(
                r"r\[u1,(\d+),#1\]", r"R\1", regex=True
            )

            # Optional: 'device'-Spalte entfernen, falls redundant
            if (
                "device" in df_signalListLF.columns
                and (df_signalListLF["device"] == df_signalListLF["path"]).all()
            ):
                df_signalListLF.drop(columns=["device"], inplace=True)

    return df_signalListLF


def import_HF_Data(data):
    """
    Importiert HF-Daten aus einem gegebenen Datensatz und formatiert sie in ein DataFrame.

    Parameters
    ----------
    data : dict
        Das Eingabedatensatz im Dictionary-Format, das die Schlüssel 'Payload' und 'Header' enthält.

    Returns
    -------
    pd.DataFrame
        Ein DataFrame mit den HF-Daten im Long-Format, inklusive Spalten für 'HFProbeCounter', 'Signal' und 'DataOrigin'.
    """
    # Initialisiere eine leere Liste für die Datensätze
    data_records = []

    for item in data["Payload"]:
        if "HFData" in item.keys():
            for i in item["HFData"]:
                # Füge das Dictionary direkt zur Liste hinzu
                data_records.append(i)
    df_HFData = pd.DataFrame(data_records)

    # Den HFDaten Namen geben
    signal_list_hf_records = []
    for item in data["Header"]["SignalListHFData"]:
        signal_list_hf_records.append(item)
    signal_namens = pd.DataFrame(signal_list_hf_records).Name.values
    df_HFData.columns = signal_namens

    # Spaltennamen umbenennen
    df_HFData = df_HFData.rename(columns={"CYCLE": "HFProbeCounter"})

    # Long Format
    df_HFData = pd.melt(df_HFData, id_vars="HFProbeCounter", var_name="Signal")

    df_HFData["DataOrigin"] = "HF_Data"

    return df_HFData


def import_LF_Data(data):
    """
    Importiert LF-Daten aus einem gegebenen Datensatz und formatiert sie in ein DataFrame.

    Parameters
    ----------
    data : dict
        Das Eingabedatensatz im Dictionary-Format, das die Schlüssel 'Payload' enthält.

    Returns
    -------
    pd.DataFrame
        Ein DataFrame mit den LF-Daten, inklusive Spalten für 'timestamp', 'Signal' und 'DataOrigin'.
    """

    data_records = []

    for item in data["Payload"]:
        if "LFData" in item.keys():
            for i in item["LFData"]:
                # Füge das Dictionary direkt zur Liste hinzu
                data_records.append(i)
    df_LFData = pd.DataFrame(data_records)

    # Konvertiere den 'timestamp' von einem String in ein datetime Objekt
    df_LFData["timestamp"] = pd.to_datetime(df_LFData["timestamp"])

    # Extrahiere den Signalnamen aus der Channelbezeichnung und lösche dieses
    df_LFData["Signal"] = df_LFData.address.str.split("/").str[3]
    # 🔧 RXXX-Umbenennung für r[u1,XXX,#1]-Signale
    df_LFData["Signal"] = df_LFData["Signal"].str.replace(
        r"r\[u1,(\d+),#1\]", r"R\1", regex=True
    )
    
    df_LFData.drop(["address"], axis=1, inplace=True)

    df_LFData["DataOrigin"] = "LF_Data"

    return df_LFData


def import_HFTimestamp_Data(data):
    """
    Importiert HFTimestamp-Daten aus einem gegebenen Datensatz und formatiert sie in ein DataFrame.

    Parameters
    ----------
    data : dict
        Das Eingabedatensatz im Dictionary-Format, das die Schlüssel 'Payload' enthält.

    Returns
    -------
    pd.DataFrame
        Ein DataFrame mit den HFTimestamp-Daten, inklusive Spalten für 'Time' und 'DataOrigin'.
    """
    # Initialisiere eine leere Liste für die Datensätze
    data_records = []

    for item in data["Payload"]:
        if "HFTimestamp" in item.keys():
            data_records.append(item["HFTimestamp"])

    df_HFTimestamp = pd.DataFrame(data_records)
    df_HFTimestamp["Time"] = pd.to_datetime(df_HFTimestamp["Time"])

    df_HFTimestamp["DataOrigin"] = "HFTimestamp_Data"

    return df_HFTimestamp


def import_HFBlockEvent(data):
    """
    Importiert HFBlockEvent-Daten aus den gegebenen Daten und bereitet einen DataFrame vor.

    Parameters
    ----------
    data : dict
        Die Daten, die die HFBlockEvent-Einträge enthalten.

    Returns
    -------
    pd.DataFrame
        Ein DataFrame, der die HFBlockEvent-Daten enthält, mit angepassten Spaltennamen.
    """

    # Initialisiere eine leere Liste für die Datensätze
    data_records = []

    for item in data["Payload"]:
        if "HFBlockEvent" in item.keys():
            data_records.append(item["HFBlockEvent"])

    # Überprüfe, ob data_records Daten enthält
    if not data_records:
        raise ValueError(
            "Keine HFBlockEvent-Daten in dem bereitgestellten Datensatz gefunden."
        )

    # Erstelle einen DataFrame aus den Datenaufzeichnungen
    df_HFBlockEvent = pd.DataFrame(data_records)

    # Überprüfe, ob der DataFrame leer ist
    if df_HFBlockEvent.empty:
        raise ValueError("Der HFBlockEvent-DataFrame ist leer.")

    # Liste der neuen Spaltennamen
    new_columns = [df_HFBlockEvent.columns[0]] + [
        "HFBlockEvent_" + col for col in df_HFBlockEvent.columns[1:]
    ]

    # Setze die neuen Spaltennamen
    df_HFBlockEvent.columns = new_columns
    
    # Um überflüssige Leerzeichen aus dem G-Code wegzubekommen.
    df_HFBlockEvent["HFBlockEvent_GCode"] = (
        df_HFBlockEvent["HFBlockEvent_GCode"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


    return df_HFBlockEvent


def create_lookup_table(df_meta, df, increment=1, interval_millisec=2):
    """
    Erstellt eine Look-Up-Tabelle mit Zeitstempeln und entsprechenden HFProbeCounter-Werten.

    Parameters
    ----------
    df_meta : pandas.DataFrame
        DataFrame, der die Schlüssel "Initial_Time" und "Initial_HFProbeCounter" enthält.
    df : pandas.DataFrame
        DataFrame, der die Spalte "HFProbeCounter" enthält.
    increment : int, optional
        Der Wert, der bei jedem Schritt zum HFProbeCounter hinzugefügt wird (default ist 1).
    interval_millisec : int, optional
        Das Zeitintervall in Millisekunden zwischen den Schritten (default ist 2).

    Returns
    -------
    pandas.DataFrame
        DataFrame, der die Zeitstempel und die entsprechenden HFProbeCounter-Werte enthält.
    """

    # Startzeit und HFProbeCounter-Werte extrahieren
    start_time = pd.to_datetime(
        df_meta.loc[df_meta.key == "Initial_Time", "value"].iloc[0]
    )
    start_HFProbeCounter = df_meta.loc[
        df_meta.key == "Initial_HFProbeCounter", "value"
    ].iloc[0]
    ende_HFProbeCounter = df.HFProbeCounter.max()

    # Berechnung der Anzahl der Schritte
    num_steps = (ende_HFProbeCounter - start_HFProbeCounter) // increment

    # Berechnung der gesamten Zeit in Millisekunden
    total_time_millisec = num_steps * interval_millisec

    # Berechnung der end_time
    end_time = start_time + pd.to_timedelta(total_time_millisec, unit="ms")

    # Erstellen der Zeitreihe mit einer Frequenz von 2 Millisekunden
    time_series = pd.date_range(start=start_time, periods=num_steps + 1, freq="2ms")

    # Erstellen der HFProbeCounter-Serie
    hf_probe_counter_series = np.arange(
        start_HFProbeCounter,
        start_HFProbeCounter + increment * (num_steps + 1),
        increment,
    )

    # Erstellen des DataFrames
    df_lookup = pd.DataFrame(
        {"Time": time_series, "HFProbeCounter": hf_probe_counter_series}
    )

    return df_lookup


def import_InsightHub_meta(file_path):
    """
    Importiert Meta-Daten aus einer Excel-Datei und formatiert die Spaltennamen.

    Parameters
    ----------
    file_path : str
        Der Pfad zur Excel-Datei, die die Meta-Daten enthält.

    Returns
    -------
    pd.DataFrame
        Ein DataFrame mit den formatierten Meta-Daten.
    """
    # Lesen der Excel-Datei
    df_InsightHub_meta = pd.read_excel(file_path)

    # Jeden String in der Liste bearbeiten, so dass nur der erste Buchstabe großgeschrieben ist
    df_InsightHub_meta.columns = [s.capitalize() for s in df_InsightHub_meta.columns]

    # Spaltennamen umbenennen
    df_InsightHub_meta.rename(columns={"Address": "Signal"}, inplace=True)

    return df_InsightHub_meta

def import_lf_variables_readable(file_path):
    """
    Importiert die LF-Variablenliste aus einer CSV-Datei mit lesbaren Steuerungsvariablen.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Der Pfad zur CSV-Datei 'LF_variables_readable.csv'.

    Returns
    -------
    pd.DataFrame
        Ein DataFrame mit allen erfolgreich gelesenen LF-Variablen und zugehörigen Metadaten.
    """
    # Lesen der CSV-Datei mit Semikolon als Trennzeichen
    df_lf = pd.read_csv(file_path, delimiter=";", encoding="utf-8")
    df_lf = df_lf[['name', 'short_description', 'nc_variable_name', 'unit']]
    
    col_dict = {'name': 'Signal', 'short_description': 'Description' , 'nc_variable_name' : 'Nc_variable_name', 'unit' : 'Unit'}
    df_lf = rename_columns(df_lf, rename_dict=col_dict)

    return df_lf

def reorder_columns(df, new_order_first_n):
    """
    Ordnet die ersten "n" Spalten eines DataFrames in einer angegebenen Reihenfolge an und behält die restlichen Spalten bei.

    Parameters
    ----------
    df : pd.DataFrame
        Der Eingabe-DataFrame, dessen Spalten neu geordnet werden sollen.
    new_order_first_n : list of str
        Eine Liste der Spaltennamen, die die gewünschte Reihenfolge der ersten "n" Spalten enthält.

    Returns
    -------
    pd.DataFrame
        Ein neuer DataFrame mit den Spalten in der gewünschten Reihenfolge.
    """
    # Alle Spalten des DataFrames
    all_columns = df.columns.tolist()

    # Erstellen der neuen Reihenfolge der Spalten
    new_order = new_order_first_n + [
        col for col in all_columns if col not in new_order_first_n
    ]

    # DataFrame neu ordnen
    df = df[new_order]

    return df


def combine_hf_lf_data(df_HFData, df_LFData):
    """
    Kombiniert HF- und LF-Daten in einen einzigen DataFrame.

    Parameters
    ----------
    df_HFData : pd.DataFrame
        DataFrame mit HF-Daten.
    df_LFData : pd.DataFrame
        DataFrame mit LF-Daten.

    Returns
    -------
    pd.DataFrame
        Kombinierter DataFrame mit HF- und LF-Daten.
    """
    return pd.concat([df_HFData, df_LFData], ignore_index=True)


def merge_with_artificial_time_series(df, df_meta):
    """
    Führt einen Left-Join des DataFrames mit künstlichen Zeitreihe durch.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, der mit Meta-Daten angereichert werden soll.
    df_meta : pd.DataFrame
        DataFrame mit Meta-Daten.

    Returns
    -------
    pd.DataFrame
        Angereicherter DataFrame.
    """
    lookup_table = create_lookup_table(df_meta, df)
    return pd.merge(df, lookup_table, on="HFProbeCounter", how="left")


def merge_with_signal_lists(df, df_signalListHF, df_signalListLF):
    """
    Führt einen Left-Join des DataFrames mit HF- und LF-Signallisten durch.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, der mit Signallisten angereichert werden soll.
    df_signalListHF : pd.DataFrame
        DataFrame mit HF-Signallisten.
    df_signalListLF : pd.DataFrame
        DataFrame mit LF-Signallisten.

    Returns
    -------
    pd.DataFrame
        Angereicherter DataFrame.
    """
    df = pd.merge(df, df_signalListHF, on="Signal", how="left")
    df = pd.merge(df, df_signalListLF, on="Signal", how="left")
    return df


def merge_with_insighthub_meta(df, df_InsightHub_meta):
    """
    Führt einen Left-Join des DataFrames mit InsightHub-Metadaten durch.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, der mit InsightHub-Metadaten angereichert werden soll.
    df_InsightHub_meta : pd.DataFrame
        DataFrame mit InsightHub-Metadaten.

    Returns
    -------
    pd.DataFrame
        Angereicherter DataFrame.
    """
    return pd.merge(df, df_InsightHub_meta, on="Signal", how="left")


def merge_asof_HFBlockEvent(df, df_HFBlockEvent):
    """
    Sortiert den DataFrame und führt einen asof-Merge mit HFBlockEvent-Daten durch.

    Parameters
    ----------
    df : pd.DataFrame
        Der zu sortierende DataFrame.
    df_HFBlockEvent : pd.DataFrame
        DataFrame mit HFBlockEvent-Daten.

    Returns
    -------
    pd.DataFrame
        Sortierter und gemergter DataFrame.
    """
    df = df.sort_values("HFProbeCounter")
    df = pd.merge_asof(df, df_HFBlockEvent, on="HFProbeCounter", direction="backward")
    df = df.sort_values(["DataOrigin", "Signal", "HFProbeCounter"], ignore_index=True)
    return df

def merge_signals_with_lfncvar(df, df_meta):
    """
    Führt ein DataFrame mit Signaldaten (`df`) mit einem Metadaten-DataFrame (`df_meta`) zusammen.

    Dabei wird:
    - der Signalname bereinigt (Teil vor '[' wird extrahiert),
    - ein Merge mit df_meta auf Basis der Spalte 'Signal' durchgeführt,
    - die Spalten 'Description' und 'Unit' intelligent zusammengeführt (combine_first),
    - überflüssige Suffix-Spalten (_x, _y) entfernt.

    Parameters
    ----------
    df : pd.DataFrame
        Das Haupt-DataFrame mit Signaldaten (z. B. Messdaten aus der CNC-Steuerung).

    df_meta : pd.DataFrame
        Das Metadaten-DataFrame mit Spalten 'Signal', 'Description', 'Unit' etc.

    Returns
    -------
    pd.DataFrame
        Ein zusammengeführtes DataFrame mit bereinigten Signalnamen und angereicherten Metadaten.
    """
    # Arbeitskopie erzeugen
    df_work = df.copy()

    # Original-Signalname sichern
    df_work['Signal_x'] = df_work['Signal']

    # Signal bereinigen (nur Teil vor erster "[")
    df_work['Signal'] = df_work['Signal'].str.split("[", n=1).str[0]

    # Merge mit Metadaten
    df_merged = pd.merge(df_work, df_meta, on="Signal", how="left")

    # Beschreibung und Einheit zusammenführen
    df_merged['Description'] = df_merged['Description_x'].combine_first(df_merged['Description_y'])
    df_merged['Unit'] = df_merged['Unit_x'].combine_first(df_merged['Unit_y'])

    # Aufräumen
    df_merged.drop(columns=['Description_x', 'Description_y', 'Unit_x', 'Unit_y', 'Signal_x'], inplace=True)

    return df_merged


def merge_dataframes(
    df_HFData,
    df_LFData,
    df_meta,
    df_signalListHF,
    df_signalListLF,
    df_InsightHub_meta,
    df_lf_NC_Var,
    df_HFBlockEvent,
):
    """
    Verarbeitet den DataFrame durch Kombination und Anreicherung mit verschiedenen Meta-Daten.

    Parameters
    ----------
    df_HFData : pd.DataFrame
        DataFrame mit HF-Daten.
    df_LFData : pd.DataFrame
        DataFrame mit LF-Daten.
    df_meta : pd.DataFrame
        DataFrame mit Meta-Daten.
    df_signalListHF : pd.DataFrame
        DataFrame mit HF-Signallisten.
    df_signalListLF : pd.DataFrame
        DataFrame mit LF-Signallisten.
    df_InsightHub_meta : pd.DataFrame
        DataFrame mit InsightHub-Metadaten.
    df_HFBlockEvent : pd.DataFrame
        DataFrame mit HFBlockEvent.

    Returns
    -------
    pd.DataFrame
        Verarbeiteter und angereicherter DataFrame.
    """
    df = combine_hf_lf_data(df_HFData, df_LFData)
    df = merge_with_signal_lists(df, df_signalListHF, df_signalListLF)
    df = merge_with_insighthub_meta(df, df_InsightHub_meta)
    df = merge_signals_with_lfncvar(df, df_lf_NC_Var)
    df = merge_asof_HFBlockEvent(df, df_HFBlockEvent)
    df = merge_with_artificial_time_series(df, df_meta)

    return df


def clean_value_type_column(df):
    """
    Aktualisiert die 'value_type'-Spalteund formatiert die 'value_type'-Spalte so, dass nur der erste Buchstabe großgeschrieben ist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, der aktualisiert und formatiert werden soll.
    df_meta : pd.DataFrame
        DataFrame, der die Meta-Informationen enthält.

    Returns
    -------
    pd.DataFrame
        Aktualisierter und formatierter DataFrame.
    """
    # Variabeln_Typ übertragen
    df.loc[df.DataOrigin == "HF_Data", "value_type"] = df.loc[
        df.DataOrigin == "HF_Data", "Type"
    ]

    # Nur der erste Buchstabe wird groß geschrieben
    df["value_type"] = df["value_type"].str.capitalize()

    return df


def add_sampling_period_HF(df, df_meta):
    """
    Fügt die Abtastrate für HF-Daten zur 'samplingPeriod'-Spalte hinzu.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, der aktualisiert werden soll.

    Returns
    -------
    pd.DataFrame
        Aktualisierter DataFrame.
    """
    # Noch die Abtastrate für HF-Daten hinzufügen
    CycleTimeMs = int(df_meta.loc[df_meta.key == "CycleTimeMs", "value"].iloc[0])
    df.loc[df.DataOrigin == "HF_Data", "samplingPeriod"] = CycleTimeMs

    return df


def clean_value_column(df):
    """
    Bereinigt die 'value'-Spalte, indem sie in numerische Werte umgewandelt wird,
    und erstellt die 'Value_string'-Spalte für nicht numerische Werte.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, der bereinigt werden soll.

    Returns
    -------
    pd.DataFrame
        Bereinigter DataFrame.
    """
    df["Value"] = pd.to_numeric(df["value"], errors="coerce")
    df["Value_string"] = df["value"].where(df["Value"].isna())
    return df


def update_groupname_for_lf_data(df):
    """
    Aktualisiert die 'Groupname'-Spalte für LF-Daten basierend auf der 'Category'-Spalte.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, der aktualisiert werden soll.

    Returns
    -------
    pd.DataFrame
        Aktualisierter DataFrame.
    """
    df.loc[df.DataOrigin == "LF_Data", "Groupname"] = df.loc[
        df.DataOrigin == "LF_Data", "Category"
    ]
    return df


def remove_and_reorder_columns(df, new_order_first_n=None):
    """
    Entfernt unnötige Spalten aus dem DataFrame und ordnet die Spalten des DataFrames in einer angegebenen Reihenfolge neu an.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, dessen Spalten neu geordnet werden sollen.
    new_order_first_n : list of str, optional
        Liste der Spaltennamen in der gewünschten Reihenfolge.
        Standardmäßig: ['Time', 'Signal', 'DataOrigin', 'Value', 'Unit', 'Value_string', 'Description', 'Groupname', 'HFBlockEvent_GCode', 'samplingPeriod', 'Axis', 'path', 'value_type'].

    Returns
    -------
    pd.DataFrame
        DataFrame mit neu geordneten Spalten.
    """
    if new_order_first_n is None:
        new_order_first_n = [
            "Time",
            "Duration_Seconds",
            "Signal",
            "DataOrigin",
            "Value",
            "Unit",
            "Value_string",
            "Description",
            "Groupname",
            "HFBlockEvent_GCode",
            "samplingPeriod",
            "Axis",
            "path",
            "value_type",
        ]

    # Entfernen der unnötigen Spalten
    df = df.drop(columns=["timestamp", "Type", "value", "Axisname", "Category"])

    # Reihenfolge der Spalten im DataFrame ändern
    all_columns = df.columns.tolist()
    new_order = new_order_first_n + [
        col for col in all_columns if col not in new_order_first_n
    ]
    df = df[new_order]

    return df


def rename_columns(df, rename_dict=None):
    """
    Benennt die Spalten eines DataFrames gemäß den angegebenen Umbenennungen um.

    Parameters
    ----------
    df : pd.DataFrame
        Der DataFrame, dessen Spalten umbenannt werden sollen.
    rename_dict : dict, optional
        Ein Wörterbuch, das die aktuellen Spaltennamen als Schlüssel und die neuen Spaltennamen als Werte enthält.
        Standardmäßig: {'Value_string': 'Value_String', 'samplingPeriod': 'SamplingPeriod', 'path': 'Path',
                        'value_type': 'Value_Type', 'id': 'ID', 'label': 'Label'}

    Returns
    -------
    pd.DataFrame
        Der DataFrame mit den umbenannten Spalten.
    """

    if rename_dict is None:
        rename_dict = {
            "Value_string": "Value_String",
            "samplingPeriod": "SamplingPeriod",
            "path": "Path",
            "value_type": "Value_Type",
            "id": "ID",
            "label": "Label",
        }

    # Filter the rename_dict to only include columns that exist in the DataFrame
    rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}

    df = df.rename(columns=rename_dict)
    return df


def convert_columns_to_category(df):
    """
    Wandelt alle Spalten eines DataFrames, die den Datentyp 'object' haben
    oder mit 'HFBlockEvent' beginnen, in den Datentyp 'category' um.

    Parameters
    ----------
    df : pandas.DataFrame
        Der Eingabe-DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Der DataFrame mit den konvertierten Spalten.

    """
    # Filtern der Spalten mit dem Datentyp 'object'
    object_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Auswählen der Spalten, die mit 'HFBlockEvent' beginnen
    hfblockevent_columns = df.filter(like="HFBlockEvent").columns.tolist()

    # Kombinieren und Entfernen von Duplikaten
    category_columns = list(set(object_columns + hfblockevent_columns))

    # Umwandeln der ausgewählten Spalten in 'category'
    for col in category_columns:
        df[col] = df[col].astype("category")

    return df


def calculate_duration_seconds(df, time_column):
    """
    Berechnet die Dauer in Sekunden ab dem frühesten Zeitstempel und fügt diese als neue Spalte in den DataFrame ein.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, der eine Spalte mit Zeitstempeln enthält.
    time_column : str
        Name der Spalte, die die Zeitstempel enthält.

    Returns
    -------
    pd.DataFrame
        DataFrame mit einer zusätzlichen Spalte für die Dauer in Sekunden.
    """
    df["Duration_Seconds"] = (
        df[time_column] - df[time_column].min()
    ).dt.total_seconds()
    return df


def calculate_wcs(
    df,
    signal_name="ENC1_POS|2",
    new_column="WCS_Y_mm",
    value_column="Value",
    signal_column="Signal",
    time_column="Duration_Seconds",
    offset=20,
    interpolate=True,
):
    """
    Berechnet die Werkstückkoordinatenposition (z. B. Y) aus einem Encodersignal
    und fügt sie zeitlich synchronisiert als neue Spalte in den gesamten DataFrame ein.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Ursprünglicher DataFrame mit allen Signalen.
    signal_name : str
        Name des Signals, das die Positionsinformation enthält (z. B. 'ENC1_POS|2').
    new_column : str
        Name der neuen Spalte für die berechnete WKS-Position (z. B. 'WCS_Y_mm').
    value_column : str, optional
        Spalte mit den Messwerten (default: 'Value').
    signal_column : str, optional
        Spalte, in der die Signalnamen stehen (default: 'Signal').
    time_column : str, optional
        Zeitspalte, über die synchronisiert wird (default: 'Duration_Seconds').
    offset : float, optional
        Konstanter Offset, der zusätzlich zum Startwert abgezogen wird (z. B. Maschinen-Nullpunkt).
    interpolate : bool, optional
        Wenn True, wird WCS_Y_mm linear auf alle Zeitpunkte interpoliert.

    Returns
    -------
    pandas.DataFrame
        DataFrame mit neuer Spalte für die Werkstückkoordinatenposition.
    """

    # Signal extrahieren
    df_pos = df.loc[df[signal_column] == signal_name, [time_column, value_column]].copy()
    df_pos = df_pos.dropna(subset=[value_column])

    if df_pos.empty:
        raise ValueError(f"Signal '{signal_name}' nicht gefunden oder leer.")

    # WKS-Position berechnen
    start_val = df_pos[value_column].iloc[0]
    df_pos[new_column] = df_pos[value_column] - start_val - offset

    # Duplikate vermeiden
    df_pos = df_pos.drop_duplicates(subset=[time_column], keep="first")

    # Interpolation aktivieren
    if interpolate:
        df_interp = df_pos[[time_column, new_column]].sort_values(by=time_column)
        df_interp = df_interp.drop_duplicates(subset=time_column).set_index(time_column)
        df_interp = df_interp.interpolate(method="linear")
        df_out = df.merge(df_interp, on=time_column, how="left")
    else:
        # Normales punktuelles Zusammenführen
        df_out = df.merge(df_pos[[time_column, new_column]], on=time_column, how="left")

    return df_out



def remove_numbers_from_axis(df, column_name="Axis"):
    """
    Entfernt alle Zahlen aus den Werten in der angegebenen Spalte des DataFrames.

    Parameters
    ----------
    df : pandas.DataFrame
        Der Eingabe-DataFrame.
    column_name : str, optional
        Der Name der Spalte, aus der die Zahlen entfernt werden sollen (default ist 'Axis').

    Returns
    -------
    df : pandas.DataFrame
        Der DataFrame mit den bearbeiteten Werten in der angegebenen Spalte.

    Examples
    --------
    >>> df = pd.DataFrame({'Axis': ['', 'X1', 'B1', 'Y1', 'Z1', 'SP1', 'X', 'Z', 'Y', np.nan]})
    >>> df = remove_numbers_from_axis(df)
    >>> print(df)
      Axis
    0
    1    X
    2    B
    3    Y
    4    Z
    5   SP
    6    X
    7    Z
    8    Y
    9  NaN
    """
    df[column_name] = df[column_name].str.replace("\d", "", regex=True)
    return df

def trim_to_HF_data_timeframe(df, time_column="Duration_Seconds", origin_column="DataOrigin", signal_column="Signal"):
    """
    Beschneidet den DataFrame so, dass nur Daten innerhalb des Zeitbereichs der HF-Daten enthalten sind.

    Parameters
    ----------
    df : pd.DataFrame
        Der Eingabe-DataFrame mit HF- und ggf. anderen Daten.
    time_column : str, optional
        Name der Spalte, die die Zeitinformationen enthält (Default: 'Duration_Seconds').
    origin_column : str, optional
        Name der Spalte, die die Datenherkunft beschreibt (Default: 'DataOrigin').
    signal_column : str, optional
        Name der Spalte, die das Signal beschreibt (Default: 'Signal').

    Returns
    -------
    pd.DataFrame
        Ein DataFrame, der nur den Zeitraum der HF-Daten enthält.
    """
    # Referenzsignal aus HF-Daten identifizieren
    ref_signal = df.loc[df[origin_column] == 'HF_Data', signal_column].unique()[0]

    # Zeitbereich des HF-Signals bestimmen
    filt = df[signal_column] == ref_signal
    time_min = df.loc[filt, time_column].min()
    time_max = df.loc[filt, time_column].max()

    # Maske auf Gesamtzeitbereich anwenden
    mask = (df[time_column] >= time_min) & (df[time_column] <= time_max)
    return df.loc[mask].reset_index(drop=True).copy()

def clean_df(df, df_meta):
    """
    Bereinigt den DataFrame, indem verschiedene Spalten aktualisiert und formatiert werden,
    unnötige Spalten entfernt und die Spalten neu geordnet werden.

    Parameters
    ----------
    df : pd.DataFrame
        Der zu bereinigende DataFrame.
    df_meta : pd.DataFrame
        DataFrame, der die Meta-Informationen enthält.

    Returns
    -------
    pd.DataFrame
        Bereinigter und neu geordneter DataFrame.
    """
    df = clean_value_type_column(df)
    df = add_sampling_period_HF(df, df_meta)
    df = clean_value_column(df)
    df = update_groupname_for_lf_data(df)
    df = calculate_duration_seconds(df, "Time")
    df = trim_to_HF_data_timeframe(df)
    df = calculate_wcs(df, signal_name="ENC1_POS|2", new_column="WCS_Y_mm", offset=20)
    df = remove_and_reorder_columns(df)
    df = remove_numbers_from_axis(df)
    df = rename_columns(df)
    return df


def tidy_data(
    dateipfad: str, dateipfad_InsightHub_meta: str = None, dateipfad_lfncvar_meta: str = None
) -> (pd.DataFrame, pd.DataFrame):
    """
    Lädt Daten aus einer JSON-Datei vom Edge Device und ergänzt sie mit Metadaten aus InsightHub.
    Gibt den bereinigten DataFrame und die Metadaten zurück.

    Parameters
    ----------
    dateipfad : str
        Pfad zur JSON-Datei mit den HF/LF-Daten.
    dateipfad_InsightHub_meta : str, optional
        Pfad zur Excel-Datei mit Insight-Hub-Metadaten. Wenn nicht angegeben,
        wird automatisch 'data/HFmeta_insight_hub.xlsx' relativ zur Projektwurzel verwendet.

    Returns
    -------
    df : pd.DataFrame
        Kombinierter und bereinigter DataFrame.
    df_meta : pd.DataFrame
        Meta-Informationen zur Messung.
    """

    # Falls kein Metapfad angegeben, automatisch auflösen
    if dateipfad_InsightHub_meta is None:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent  # geht davon aus, dass Datei in src/ liegt
        dateipfad_InsightHub_meta = project_root / "config" / "HFmeta_insight_hub.xlsx"

    if dateipfad_lfncvar_meta is None:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent  # geht davon aus, dass Datei in src/ liegt
        dateipfad_lfncvar_meta = project_root / "config" / "LF_variables_readable.csv"
    # Debug-Ausgabe (optional entfernen)
    # print(f"→ Lade Insight-Hub-Metadaten von: {dateipfad_InsightHub_meta}")

    # Daten laden
    data = import_data(dateipfad)
    df_InsightHub_meta = import_InsightHub_meta(dateipfad_InsightHub_meta)
    df_lf_NC_Var = import_lf_variables_readable(dateipfad_lfncvar_meta)

    df_meta = import_meta_df(data)
    df_signalListHF = import_signalListHF(data)
    df_signalListLF = import_signalListLF(data)

    df_HFData = import_HF_Data(data)
    df_LFData = import_LF_Data(data)
    df_HFBlockEvent = import_HFBlockEvent(data)

    # Zusammenführen und Bereinigen
    df = merge_dataframes(
        df_HFData,
        df_LFData,
        df_meta,
        df_signalListHF,
        df_signalListLF,
        df_InsightHub_meta,
        df_lf_NC_Var,
        df_HFBlockEvent,
    )

    df = clean_df(df, df_meta)

    return df, df_meta
