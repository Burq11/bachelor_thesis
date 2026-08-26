# viz/widgets_digital_twin.py

import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
from pathlib import Path
import re

def show_plattenauswahl_widget(processed_path, summarize_fn, plot_fn, analyze_fn, plate_state: dict = None):
    """
    Interaktives Widget zur Auswahl einer Fräsplatte mit Chatter-Visualisierung.

    Funktion:
    - Zeigt Dropdown zur Auswahl aller verfügbaren Platten
    - Lädt und aggregiert alle zugehörigen .parquet-Dateien
    - Visualisiert die Segmentierung mit plot_fn
    - Setzt die globale Variable `platte`
    - Optional: speichert fig, df_summary, platte in einem übergebenen Zustand

    Parameter
    ----------
    processed_path : Path
        Pfad zum Ordner mit allen 'Platte_*_Nut_*.parquet'-Dateien.
    summarize_fn : function
        Funktion zur Aggregation pro Nut-DataFrame. Erwartet einen df, gibt df_summary zurück.
    plot_fn : function
        Funktion zur Visualisierung der gesamten Platte. Erwartet df_summary.
    analyze_fn : function
        Kombiniert das Laden, Aggregieren und Visualisieren für eine Platte.
        Signatur: (platte, processed_path, summarize_fn, plot_fn) → (df_summary, fig)
    plate_state : dict, optional
        Optionaler Container zur Speicherung von:
        - plate_state["platte"] → Platten-ID
        - plate_state["df_summary"] → aggregierte DataFrame
        - plate_state["fig"] → Visualisierung
    """
    global platte
    output_platte = widgets.Output()

    parquet_files = list(processed_path.glob("Platte_*_Nut_*.parquet"))
    platten_ids = sorted({int(f.name.split("_")[1]) for f in parquet_files})

    dropdown = widgets.Dropdown(
        options=platten_ids,
        description="Platte:",
        layout=widgets.Layout(width="200px")
    )

    def on_platte_change(change):
        global platte
        output_platte.clear_output()
        platte = change["new"]

        df_summary, fig = analyze_fn(platte, processed_path, summarize_fn, plot_fn)

        # In Zustand speichern, falls gewünscht
        if plate_state is not None:
            plate_state["platte"] = platte
            plate_state["df_summary"] = df_summary
            plate_state["fig"] = fig

        with output_platte:
            fig.show()

    dropdown.observe(on_platte_change, names="value")
    display(widgets.VBox([dropdown, output_platte]))
    dropdown.value = dropdown.options[0]  # Triggert Initialisierung

import re

import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output, display

from pathlib import Path


def show_nutauswahl_widget(processed_path: Path, state: dict = None):
    """
    Interaktives Dropdown-Widget zur Auswahl einer Nut und Anzeige von Dateiinformationen.

    Diese Funktion:
    - Listet alle Nuten der aktuell gewählten Platte (globale Variable 'platte' erforderlich)
    - Lädt die zugehörige .parquet-Datei bei Auswahl
    - Zeigt Basisinformationen zur Datei (Anzahl Zeilen, Spalten, Spaltennamen)
    - Optional: speichert das geladene DataFrame unter `state["df"]`

    Parameter
    ----------
    processed_path : Path
        Pfad zum Ordner mit den vorverarbeiteten .parquet-Dateien (pro Nut)
    state : dict, optional
        Optionaler Container zum Speichern des geladenen DataFrames.
        Wenn übergeben, wird `state["df"]`, `state["nut"]` und `state["path"]` gesetzt.
    """
    output_nut = widgets.Output()

    # Platten- und Nuten-Mapping erstellen
    parquet_files = list(processed_path.glob("Platte_*_Nut_*.parquet"))
    platten_nuten_map = {}
    for f in parquet_files:
        match = re.search(r"Platte_(\d+).*?Nut_(\d+)", f.name)
        if match:
            p = int(match.group(1))
            n = int(match.group(2))
            platten_nuten_map.setdefault(p, {})[n] = f

    # Voraussetzung: globale Variable 'platte' muss gesetzt sein
    if "platte" not in globals():
        print("⚠️ Bitte zuerst eine Platte auswählen.")
        return

    nuten = sorted(platten_nuten_map.get(platte, {}).keys())
    if not nuten:
        print(f"⚠️ Keine Nuten für Platte {platte} gefunden.")
        return

    # Dropdown-Widget
    nut_dropdown = widgets.Dropdown(
        options=nuten,
        description='Nut:',
        layout=widgets.Layout(width='200px')
    )

    def on_nut_change(change):
        output_nut.clear_output()
        nut = change["new"]
        path = platten_nuten_map[platte][nut]
        df_local = pd.read_parquet(path)

        # Zustand speichern
        if state is not None:
            state["df"] = df_local
            state["nut"] = nut
            state["path"] = path

        with output_nut:
            print(f"📄 Datei: {path.name}")
            print(f"🔢 Zeilen: {len(df_local):,}  |  Spalten: {df_local.shape[1]}")

    nut_dropdown.observe(on_nut_change, names="value")
    display(widgets.VBox([nut_dropdown, output_nut]))

    # Initialauswahl setzen
    nut_dropdown.value = nut_dropdown.options[0]


def show_plate_widget(processed_path, plate_state=None):
    """
    Komfort-Wrapper zur Anzeige des Chatter-Digital-Twins für eine Platte.

    Intern wird `show_plattenauswahl_widget(...)` aufgerufen mit:
    - summarize_chatter_cases
    - plot_digital_twin_segmented
    - analyze_platte

    Parameter
    ----------
    processed_path : Path
        Pfad zum Ordner mit 'Platte_*_Nut_*.parquet'-Dateien
    plate_state : dict, optional
        Optionaler Zustand, in dem platte, df_summary und fig gespeichert werden.
    """
    from validation_data_access.legacy.Oxford.src.data_processing import summarize_chatter_cases, analyze_platte
    from validation_data_access.legacy.Oxford.viz.visualizer import plot_digital_twin_segmented
    from validation_data_access.legacy.Oxford.viz.widgets_digital_twin import show_plattenauswahl_widget

    show_plattenauswahl_widget(
        processed_path=processed_path,
        summarize_fn=summarize_chatter_cases,
        plot_fn=plot_digital_twin_segmented,
        analyze_fn=analyze_platte,
        plate_state=plate_state
    )

#########HEATMAP###############
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def show_heatmap_widget(processed_path, heatmap_state=None):
    """
    Interactive widget for digital twin heatmap visualization with spatial color gradients.
    Final version – colorbar shows true physical vibration amplitude (cutting region only),
    and normalized amplitude (0–1) for direct comparison to normalized plots.
    """
    global platte
    output_platte = widgets.Output()
    if heatmap_state is None:
        heatmap_state = {}

    # Find available plates 
    parquet_files = list(processed_path.glob("Platte_*_Nut_*.parquet"))
    platten_ids = sorted({int(f.name.split("_")[1]) for f in parquet_files})
    
    if not platten_ids:
        print("No plate files found!")
        return
    
    # Widgets 
    dropdown_platte = widgets.Dropdown(
        options=platten_ids,
        value=platten_ids[0],
        description="Platte:",
        layout=widgets.Layout(width="200px"),
        style={'description_width': 'initial', 'font_family': 'Arial'}
    )
    
    slider_bin = widgets.SelectionSlider(
        options=list(range(2, 13)),   # 2..12 mm
        value=10,
        description="Y-Bin [mm]:",
        continuous_update=False,
        layout=widgets.Layout(width="400px"),
        style={"description_width": "80px", "font_family": "Arial"}
    )

    slider_qw = widgets.SelectionSlider(
        options=[10000, 20000, 50000],
        value=50000,
        description="Qw step [mm³/min]:",
        continuous_update=False,
        layout=widgets.Layout(width="450px"),
        style={"description_width": "140px", "font_family": "Arial"}
    )
    slider_qw.disabled = True  # enable only after heatmap is generated
    
    def redraw_qw_overlay(fig, df_summary, plate_height=245, qw_step=50000):
        """
        Update ONLY Qw iso-lines (dotted line shapes) on an existing heatmap figure.
        No heatmap recomputation.
        """
        import numpy as np
        from plotly.colors import sample_colorscale
    
        if df_summary is None or "Drehzahl" not in df_summary.columns:
            return fig
    
        # 1) Remove previous dotted Qw line shapes
        old_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        kept_shapes = []
        for s in old_shapes:
            # Qw shapes are 'line' with dash='dot' in your heatmap generator
            if getattr(s, "type", None) == "line" and getattr(getattr(s, "line", None), "dash", None) == "dot":
                continue
            kept_shapes.append(s)
        fig.layout.shapes = tuple(kept_shapes)
    
        # 2) Rebuild Qw lines (same logic as in visualizer.py, but with configurable step)
        a_e = 10.0       # mm
        f_rev = 0.18     # mm/rev
        k = a_e * f_rev  # 1.8
    
        ap_start = 2.0
        ap_end = 10.56
    
        # Slot positions from summary if available
        slot_positions = {}
        if "Nut_ID" in df_summary.columns and "X_Position_Nut" in df_summary.columns:
            slot_positions = df_summary.set_index("Nut_ID")["X_Position_Nut"].to_dict()
    
        slots = sorted(slot_positions.keys()) if slot_positions else sorted(df_summary["Nut_ID"].dropna().unique())
    
        slot_rpm = {}
        for slot_id in slots:
            try:
                rpm_val = float(df_summary.loc[df_summary["Nut_ID"] == slot_id, "Drehzahl"].iloc[0])
                slot_rpm[slot_id] = rpm_val
            except Exception:
                continue
    
        if not slot_rpm:
            return fig
    
        max_rpm = max(slot_rpm.values())
        qw_max = k * ap_end * max_rpm  # mm³/min
    
        level_max = int(np.floor(qw_max / float(qw_step))) * int(qw_step)
        if level_max < qw_step:
            return fig
    
        qw_levels = list(range(int(qw_step), int(level_max) + int(qw_step), int(qw_step)))
    
        # colorscale similar to current grey–blue fade
        IWF_GreyBlue_fade_scale = [[0.0, "#dfe7ec"], [0.5, "#9fb6c4"], [1.0, "#3a515f"]]
        positions = [0.5] if len(qw_levels) <= 1 else [i / (len(qw_levels) - 1) for i in range(len(qw_levels))]
        qw_colors = sample_colorscale(IWF_GreyBlue_fade_scale, positions)
    
        # draw dotted line segments, layer="above"
        for qw_level, col in zip(qw_levels, qw_colors):
            x_line = []
            y_line = []
    
            for slot_id in slots:
                rpm = slot_rpm.get(slot_id, None)
                if rpm is None:
                    x_line.append(None); y_line.append(None)
                    continue
    
                ap_target = qw_level / (k * rpm)
                if ap_start <= ap_target <= ap_end:
                    y_target = plate_height * (ap_target - ap_start) / (ap_end - ap_start)
                    x_target = slot_positions.get(slot_id, 10 + slot_id * 15)
                    x_line.append(x_target); y_line.append(y_target)
                else:
                    x_line.append(None); y_line.append(None)
    
            for i in range(len(x_line) - 1):
                x0, x1 = x_line[i], x_line[i + 1]
                y0, y1 = y_line[i], y_line[i + 1]
                if x0 is None or x1 is None or y0 is None or y1 is None:
                    continue
    
                fig.add_shape(
                    type="line",
                    x0=x0, y0=y0,
                    x1=x1, y1=y1,
                    line=dict(color=col, width=2, dash="dot"),
                    layer="above",
                )
    
        return fig

    update_button = widgets.Button(
        description="Generate Heatmap",
        button_style='',
        layout=widgets.Layout(width="150px")
    )
    update_button.style.button_color = '#990000'
    update_button.style.text_color = 'white'
    update_button.style.font_family = 'Arial'
    update_button.style.font_weight = 'bold'
    
    # Interaction handlers 
    def mark_interaction(change):
        output_platte.clear_output()
        with output_platte:
            print(f"Settings: Plate {dropdown_platte.value}, Bin size {slider_bin.value} mm")
            print("Click 'Generate Heatmap' to create plot")
    
    def update_heatmap(button=None):
        global platte
        output_platte.clear_output()
    
        platte = dropdown_platte.value
        bin_size_mm = slider_bin.value
    
        with output_platte:
            print(f"Creating heatmap for Plate {platte} with bin size {bin_size_mm} mm...")
    
            try:
                from validation_data_access.legacy.Oxford.src.data_processing import prepare_equal_bins_heatmap, analyze_platte, summarize_chatter_cases
                from validation_data_access.legacy.Oxford.viz.visualizer import plot_digital_twin_heatmap_gradient
    
                parquet_files = sorted(processed_path.glob(f"Platte_{platte}_*Nut_*.parquet"))
                if not parquet_files:
                    print(f"No files found for Plate {platte}")
                    return
    
                # Build heatmap data 
                all_heatmap_data = []
                y_ranges = []  # to store cutting-region Y ranges
                for file_path in parquet_files:
                    df = pd.read_parquet(file_path)
                    df_heatmap = prepare_equal_bins_heatmap(df, bin_size_mm=bin_size_mm)
                    if not df_heatmap.empty:
                        all_heatmap_data.append(df_heatmap)
                        y_ranges.append((df_heatmap["Y_min"].min(), df_heatmap["Y_max"].max()))
    
                if not all_heatmap_data:
                    print(f"No valid heatmap data for Plate {platte}")
                    return
    
                df_heatmap = pd.concat(all_heatmap_data, ignore_index=True)
    
                # Compute global normalization 
                vmin, vmax = df_heatmap["RMS_raw"].min(), df_heatmap["RMS_raw"].max()
                if vmax > vmin:
                    df_heatmap["RMS_normalized_global"] = (df_heatmap["RMS_raw"] - vmin) / (vmax - vmin)
                else:
                    df_heatmap["RMS_normalized_global"] = 0.0
                
                # Compute true vibration amplitude for RMS_min and RMS_max bins (cutting region only)
              
                #   RMS = 0 → represent the *most stable* bin        (lowest RMS_raw)
                #   RMS = 1 → represent the *worst chatter* bin      (highest RMS_raw)
                
                #   For each of these two bins, we want to extract the *actual*
                #   oscilloscope peak vibration amplitude (from the raw X-signal)
        
                # Column identifying the slot in df_heatmap (Nut or Nut_ID depending on your files)
                slot_col = "Nut" if "Nut" in df_heatmap.columns else "Nut_ID"
        
                # Locate the bins with globally smallest and largest RMS_raw 
                idx_rms_min = df_heatmap["RMS_raw"].idxmin()  # → index of most stable bin
                idx_rms_max = df_heatmap["RMS_raw"].idxmax()  # → index of chatter hotspot
        
                # Extract the full row for both bins
                bin_min = df_heatmap.loc[idx_rms_min]  # row with smallest RMS_raw
                bin_max = df_heatmap.loc[idx_rms_max]  # row with largest RMS_raw
        
                # Slot numbers for those bins
                slot_min = int(bin_min[slot_col])
                slot_max = int(bin_max[slot_col])
        
                # Y-limits of the most stable bin
                y_min_min = float(bin_min["Y_min"])
                y_max_min = float(bin_min["Y_max"])
        
                # Y-limits of the worst chatter bin
                y_min_max = float(bin_max["Y_min"])
                y_max_max = float(bin_max["Y_max"])
        
                # Lists to collect X-axis oscillation values belonging to each bin
                vals_min = []   # raw vibrations in RMS_min bin
                vals_max = []   # raw vibrations in RMS_max bin
        
                # Loop over all parquet files belonging to this plate 
                # Each file corresponds to exactly one slot (Nut)
                for file_path in parquet_files:
                    df = pd.read_parquet(file_path)
        
                    # Determine which slot this file belongs to
                    if slot_col in df.columns:
                        slot_here = int(df[slot_col].iloc[0])
                    else:
                        # Fallback: if slot is not encoded per file (rare)
                        slot_here = None
        
                    # Extract only oscilloscope X-channel data
                    df_sig = df[
                        (df["Axis"] == "X") &
                        (df["DataOrigin"] == "Oscilloscope")
                    ]
        
                    # OPTIONAL: remove extreme spikes if desired
                    # df_sig = df_sig[df_sig["Value"].between(-1.0, 1.0)]
        
                    # If this file corresponds to the RMS_min bin slot:
                    if slot_here == slot_min:
                        # Select oscilloscope samples exactly within this bin's Y-range
                        mask_min = (
                            (df_sig["WCS_Y_mm"] >= y_min_min) &
                            (df_sig["WCS_Y_mm"] <= y_max_min)
                        )
                        vals_min.extend(df_sig.loc[mask_min, "Value"].values)
        
                    # If this file corresponds to the RMS_max bin slot:
                    if slot_here == slot_max:
                        # Select oscilloscope samples exactly within this bin's Y-range
                        mask_max = (
                            (df_sig["WCS_Y_mm"] >= y_min_max) &
                            (df_sig["WCS_Y_mm"] <= y_max_max)
                        )
                        vals_max.extend(df_sig.loc[mask_max, "Value"].values)
        
                # Extract final amplitudes 
                if vals_min:
                    amp_min = float(np.min(vals_min))   # often ~ small periodic vibration
                else:
                    amp_min = 0.0                       # fallback if no data found
        
                if vals_max:
                    amp_max = float(np.max(vals_max))   # chatter spikes → large values
                else:
                    amp_max = 0.0                       # fallback
        
                # These amplitudes correspond directly to RMS=0 and RMS=1 regions
                true_min = amp_min
                true_max = amp_max
                    
                # Summary and plot 
                df_summary, _ = analyze_platte(platte, processed_path, summarize_chatter_cases, lambda x: None)
                fig = plot_digital_twin_heatmap_gradient(df_heatmap, df_summary=df_summary)
    
                # Custom colorbar label 
                colorbar_label = (
                    f"<b>Vibration Intensity</b><br><br><br>"
                    f"<b>True X-vibration min–max mapped to RMS:</b><br>"
                    f"min: {true_min:.3f} – "
                    f"max: {true_max:.3f}"
                )

                for trace in fig.data:
                    if hasattr(trace, "colorbar") and trace.colorbar is not None:
                        trace.colorbar.title = dict(
                            text=colorbar_label,
                            side="right",
                            font=dict(size=11, color="black")
                        )
    
                # Enable Qw slider now that a figure exists
                slider_qw.disabled = False
                
                # Redraw overlay once immediately, so default Qw lines get replaced
                fig = redraw_qw_overlay(fig, df_summary=df_summary, plate_height=245, qw_step=slider_qw.value)
                
                # Save state (save the updated fig)
                if heatmap_state is not None:
                    heatmap_state.update({
                        "platte": platte,
                        "bin_size_mm": bin_size_mm,
                        "df_heatmap": df_heatmap,
                        "df_summary": df_summary,
                        "fig": fig,
                        "method": "spatial_heatmap"
                    })
                              
                # Display 
                slot_column = 'Nut' if 'Nut' in df_heatmap.columns else 'Nut_ID'
                n_slots = df_heatmap[slot_column].nunique()
                n_segments = len(df_heatmap)
                print(f"Heatmap created: {n_slots} slots, {n_segments} segments")
                display(fig)
    
            except Exception as e:
                print(f"Error creating heatmap: {e}")
                
    def on_qw_change(change):
        if "fig" not in heatmap_state or "df_summary" not in heatmap_state:
            return
    
        fig = heatmap_state["fig"]
        df_summary = heatmap_state["df_summary"]
    
        # Update overlay only
        fig = redraw_qw_overlay(fig, df_summary=df_summary, plate_height=245, qw_step=change["new"])
        heatmap_state["fig"] = fig
    
        output_platte.clear_output()
        with output_platte:
            display(fig)

    slider_qw.observe(on_qw_change, names="value")

    # Event bindings
    dropdown_platte.observe(mark_interaction, names="value")
    slider_bin.observe(mark_interaction, names="value")
    update_button.on_click(update_heatmap)
    
    # Layout 
    display(widgets.VBox([
        widgets.HBox([dropdown_platte, slider_bin, update_button]),
        slider_qw,
        output_platte
    ]))
    
    with output_platte:
        print("Select plate and bin size above, then click 'Generate Heatmap'")
        print(f"Current settings: Plate {dropdown_platte.value}, Bin size {slider_bin.value} mm")

#########PCA###############
import ipywidgets as widgets
from IPython.display import display
from pathlib import Path

def show_plate_selector(processed_path, plate_state=None):
    """
    Minimal Plate Selector (no visualization).
    Just sets global variable 'platte' and saves it into plate_state.
    """
    global platte
    parquet_files = list(processed_path.glob("Platte_*_Nut_*.parquet"))
    platten_ids = sorted({int(f.name.split("_")[1]) for f in parquet_files})

    dropdown = widgets.Dropdown(
        options=platten_ids,
        description="Platte:",
        layout=widgets.Layout(width="200px")
    )

    output = widgets.Output()

    def on_change(change):
        global platte
        platte = change["new"]
        output.clear_output()
        with output:
            print(f"✅ Selected Plate: {platte}")
        if plate_state is not None:
            plate_state["platte"] = platte

    dropdown.observe(on_change, names="value")
    display(widgets.VBox([dropdown, output]))
    dropdown.value = dropdown.options[0]  # triggers initial value
