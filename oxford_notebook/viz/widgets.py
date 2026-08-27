"""ipywidgets front end for the digital-twin notebooks.

Renamed from `viz/widgets_digital_twin.py` in Martin Heper's (M.Sc., IWF, TU Berlin)
Parquet-based analysis system, preserved as received at
validation_data_access/legacy/Oxford/viz/widgets_digital_twin.py. Every function below
carries a provenance note recording whether it is unchanged, adapted, or new in this
thesis.

The legacy `show_plattenauswahl_widget`, `show_nutauswahl_widget`, `show_plate_widget`
and `show_plate_selector` were replaced by the two combined selection widgets below,
which read through `src.provider` instead of scanning a processed-data directory.
"""


import builtins
import time

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from plotly.colors import sample_colorscale

from src import provider
from src.data_processing import (
    prepare_equal_bins_heatmap_sql,
    get_min_max_amplitudes_sql_from_db,
    summarize_chatter_cases_sql,
    filter_constant_HF_signals,
)
from viz.visualizer import (
    create_axiswise_plots2,
    plot_digital_twin_heatmap_gradient,
)


_ACTIVE_HEATMAP_WIDGET = None


# Provenance: new in this thesis.
def _close_widget(widget):
    if widget is None:
        return
    try:
        widget.close()
    except Exception:
        pass

# Provenance: new in this thesis. Replaces the legacy two-step
# `show_plattenauswahl_widget` / `show_nutauswahl_widget` pair.
def show_plate_slot_selection_widget(default_plate=None, state=None):
    plates = provider.plates()
    if not plates:
        display(widgets.HTML("<b>No plates found in provider.</b>"))
        return

    plate_dropdown = widgets.Dropdown(
        options=plates,
        value=default_plate if default_plate in plates else plates[0],
        description="Plate:",
        layout=widgets.Layout(width="200px")
    )

    slot_dropdown = widgets.Dropdown(
        options=provider.slots(plate_dropdown.value),
        description="Slot:",
        layout=widgets.Layout(width="200px")
    )

    confirm_button = widgets.Button(
        description="Load DataFrame",
        button_style='success',
        layout=widgets.Layout(width="150px")
    )
    style_button(confirm_button)
    
    output = widgets.Output()

    def update_slots(change):
        slots = provider.slots(change["new"])
        slot_dropdown.options = slots if slots else []
        if slots:
            slot_dropdown.value = slots[0]

    plate_dropdown.observe(update_slots, names="value")

    def on_confirm(b):
        output.clear_output()
        plate = plate_dropdown.value
        slot = slot_dropdown.value
        with output:
            print(f"Loading DataFrame for Plate {plate}, Slot {slot} ...")
            df = provider.df(plate, slot)
            if df is not None and not df.empty:
                builtins.selected_df = df
                print(f"✅ DataFrame for Plate {plate}, Slot {slot} loaded as 'selected_df' (rows: {len(df)}, columns: {df.shape[1]})")
                if state is not None:
                    state["plate"] = plate
                    state["slot"] = slot
                    state["df"] = df
            else:
                print(f"❌ No data found for Plate {plate}, Slot {slot}.")

    confirm_button.on_click(on_confirm)

    display(widgets.VBox([widgets.HBox([plate_dropdown, slot_dropdown, confirm_button]), output]))
    

#########HEATMAP###############  BASE IMPLEMENTATION BY OTHER STUDENTS
# Provenance: adapted from the legacy system — the overlay and bin inputs now come
# from `src.provider` and the SQL helpers instead of per-slot Parquet loads, and
# staged progress reporting was added. The QW overlay drawing is unchanged.
def show_heatmap_widget(heatmap_state=None):
    """
    Interactive widget for digital twin heatmap visualization with spatial color gradients.
    Final version – colorbar shows true physical vibration amplitude (cutting region only),
    and normalized amplitude (0–1) for direct comparison to normalized plots.
    """
    global platte, _ACTIVE_HEATMAP_WIDGET

    clear_output(wait=True)

    if _ACTIVE_HEATMAP_WIDGET is not None:
        _close_widget(_ACTIVE_HEATMAP_WIDGET)
        _ACTIVE_HEATMAP_WIDGET = None

    output_platte = widgets.Output()
    if heatmap_state is None:
        heatmap_state = {}

    # Find available plates 
    
    platten_ids = provider.plates()
    
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
    qw_status = widgets.HTML("")

    progress_text = widgets.HTML("<b>Heatmap progress:</b> waiting to start")
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=4,
        description="Progress:",
        bar_style="info",
        layout=widgets.Layout(width="420px"),
    )
    progress_box = widgets.VBox([progress_text, progress_bar])

    def set_progress(step, message):
        progress_bar.value = step
        progress_text.value = f"<b>Heatmap progress:</b> {message}"
    
    def redraw_qw_overlay(fig, df_summary, plate_height=245, qw_step=50000):
        """
        Update ONLY Qw iso-lines (dotted line shapes) on an existing heatmap figure.
        No heatmap recomputation.
        """
        if df_summary is None or "Drehzahl" not in df_summary.columns:
            return fig
    
        # 1) Remove previous dotted Qw line shapes
        old_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        kept_shapes = []
        for s in old_shapes:
            # Qw shapes are dotted overlay paths in the current heatmap generator.
            if getattr(getattr(s, "line", None), "dash", None) == "dot" and getattr(s, "type", None) in {"line", "path"}:
                continue
            kept_shapes.append(s)
        fig.layout.shapes = tuple(kept_shapes)
    
        # 2) Rebuild Qw lines (same logic as in visualizer.py, but with configurable step)
        a_e = 10.0       # mm
        f_rev = 0.18     # mm/rev
        k = a_e * f_rev  # 1.8
    
        ap_start = 2.0
        ap_end = 10.56
    
        # Slot positions from summary if available.
        # Safeguard: heatmap bins are keyed by Nut, while legacy overlays often key by Nut_ID.
        slot_positions_by_nutid = {}
        slot_positions_by_nut = {}

        if "X_Position_Nut" in df_summary.columns:
            if "Nut_ID" in df_summary.columns:
                slot_positions_by_nutid = df_summary.set_index("Nut_ID")["X_Position_Nut"].to_dict()
            if "Nut" in df_summary.columns:
                slot_positions_by_nut = df_summary.set_index("Nut")["X_Position_Nut"].to_dict()

        def _slot_pos(slot_id):
            for key in (slot_id, float(slot_id) if slot_id is not None else slot_id):
                if key in slot_positions_by_nutid:
                    return slot_positions_by_nutid[key]
                if key in slot_positions_by_nut:
                    return slot_positions_by_nut[key]
            try:
                ikey = int(slot_id)
            except Exception:
                return None
            for key in (ikey, float(ikey)):
                if key in slot_positions_by_nutid:
                    return slot_positions_by_nutid[key]
                if key in slot_positions_by_nut:
                    return slot_positions_by_nut[key]
            return None

        # Prefer Nut_ID ordering if present (preserves legacy visuals)
        if "Nut_ID" in df_summary.columns:
            slots = sorted(df_summary["Nut_ID"].dropna().unique())
        elif "Nut" in df_summary.columns:
            slots = sorted(df_summary["Nut"].dropna().unique())
        else:
            slots = []
    

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

        # Build arrays once, then stage path shapes and attach them in one batch.
        slot_ids = np.array(slots)
        x_array = np.empty(len(slot_ids), dtype=float)
        rpm_array = np.full(len(slot_ids), np.nan, dtype=float)

        for i, slot_id in enumerate(slot_ids):
            rpm_array[i] = float(slot_rpm.get(slot_id, np.nan)) if slot_id in slot_rpm else np.nan
            x_target = _slot_pos(slot_id)
            if x_target is None:
                try:
                    x_target = 10 + float(slot_id) * 15
                except Exception:
                    x_target = 10.0
            x_array[i] = float(x_target)

        valid_rpm_mask = np.isfinite(rpm_array) & (rpm_array > 0)
        qw_shapes = []
        qw_shape_count = 0

        for qw_level, col in zip(qw_levels, qw_colors):
            with np.errstate(divide="ignore", invalid="ignore"):
                ap_target = qw_level / (k * rpm_array)

            mask = valid_rpm_mask & (ap_target >= ap_start) & (ap_target <= ap_end)
            if not mask.any():
                continue

            y_target = plate_height * (ap_target - ap_start) / (ap_end - ap_start)
            idx = np.where(mask)[0]
            breaks = np.where(np.diff(idx) != 1)[0]
            starts = np.concatenate(([idx[0]], idx[breaks + 1]))
            ends = np.concatenate((idx[breaks], [idx[-1]]))

            for start_idx, end_idx in zip(starts, ends):
                path_parts = [f"M {x_array[start_idx]:.2f},{y_target[start_idx]:.2f}"]
                for j in range(start_idx + 1, end_idx + 1):
                    path_parts.append(f"L {x_array[j]:.2f},{y_target[j]:.2f}")
                qw_shapes.append(
                    dict(
                        type="path",
                        path=" ".join(path_parts),
                        line=dict(color=col, width=2, dash="dot"),
                        layer="above",
                    )
                )
                qw_shape_count += 1

        if qw_shapes:
            existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
            existing_shapes.extend(qw_shapes)
            fig.update_layout(shapes=existing_shapes)
    
        return fig

    update_button = widgets.Button(
        description="Generate Heatmap",
        button_style='',
        layout=widgets.Layout(width="150px")
    )
    style_button(update_button)
    
    # Interaction handlers 
    def mark_interaction(change):
        output_platte.clear_output()
        with output_platte:
            print(f"Settings: Plate {dropdown_platte.value}, Bin size {slider_bin.value} mm")
            print("Click 'Generate Heatmap' to create plot")
    
    def update_heatmap(button=None):
        global platte
        output_platte.clear_output()

        t0 = time.perf_counter()

        set_progress(0, "starting")
    
        platte = dropdown_platte.value
        bin_size_mm = slider_bin.value
    
        with output_platte:
            display(progress_box)
            set_progress(1, f"creating heatmap for Plate {platte} with bin size {bin_size_mm} mm")
            print(f"Creating heatmap for Plate {platte} with bin size {bin_size_mm} mm...")
    
            try:
                # Get heatmap data from SQL
                df_heatmap = prepare_equal_bins_heatmap_sql(
                    platte,
                    bin_size_mm=bin_size_mm,
                    compute_normalized_global=True,
                )
                set_progress(2, "loaded heatmap data")

                if df_heatmap.empty:
                    print(
                        f"No heatmap data for Plate {platte}: This is a known data gap (no connected accelerometer"
                        f"in this recording)"
                    )
                    return

                # Global normalization (prefer SQL result; fallback to Python)
                if "RMS_normalized_global" not in df_heatmap.columns:
                    vmin, vmax = df_heatmap["RMS_raw"].min(), df_heatmap["RMS_raw"].max()
                    if vmax > vmin:
                        df_heatmap["RMS_normalized_global"] = (df_heatmap["RMS_raw"] - vmin) / (vmax - vmin)
                    else:
                        df_heatmap["RMS_normalized_global"] = 0.0
                
                # Get min/max amplitudes from SQL (extrema bin selection in DuckDB)
                true_min, true_max = get_min_max_amplitudes_sql_from_db(
                    platte,
                    bin_size_mm=bin_size_mm,
                    target_signal="X",
                    target_origin="ET200_Data",
                )

                # Summary and plot (DuckDB-first; avoids per-slot raw DataFrame loads)
                df_summary = summarize_chatter_cases_sql(platte)

                fig = plot_digital_twin_heatmap_gradient(
                    df_heatmap,
                    df_summary=df_summary,
                )
                set_progress(3, "building figure and overlays")
    
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
                set_progress(4, "done")
                print(f"Heatmap created: {n_slots} slots, {n_segments} segments")
                display(fig)
    
            except Exception as e:
                print(f"Error creating heatmap: {e}")
                
    def on_qw_change(change):
        if "fig" not in heatmap_state or "df_summary" not in heatmap_state:
            return
    
        fig = heatmap_state["fig"]
        df_summary = heatmap_state["df_summary"]
    
        qw_status.value = f"<b>Qw step:</b> applying {change['new']:,} mm³/min ..."

        # Update overlay only
        fig = redraw_qw_overlay(fig, df_summary=df_summary, plate_height=245, qw_step=change["new"])
        heatmap_state["fig"] = fig

        qw_status.value = f"<b>Qw step:</b> applied {change['new']:,} mm³/min"
    
        output_platte.clear_output()
        with output_platte:
            display(fig)

    slider_qw.observe(on_qw_change, names="value")

    # Event bindings
    dropdown_platte.observe(mark_interaction, names="value")
    slider_bin.observe(mark_interaction, names="value")
    update_button.on_click(update_heatmap)
    
    # Layout 
    root_widget = widgets.VBox([
        widgets.HBox([dropdown_platte, slider_bin, update_button]),
        slider_qw,
        qw_status,
        output_platte
    ])

    heatmap_state["widget"] = root_widget
    _ACTIVE_HEATMAP_WIDGET = root_widget
    display(root_widget)
    
    with output_platte:
        print("Select plate and bin size above, then click 'Generate Heatmap'")
        print(f"Current settings: Plate {dropdown_platte.value}, Bin size {slider_bin.value} mm")

# Utility Voila: Button to trigger plot generation for the selected DataFrame.
# Provenance: new in this thesis.
def show_generate_plots_button(get_df_func=None, get_plot_dfs_func=None):
    """
    Displays a button that generates plots for the selected DataFrame.
    get_df_func: Optional function that returns the DataFrame to plot (default: uses global 'selected_df').
    get_plot_dfs_func: Optional function that returns pre-split DataFrames for External/HF/LF plots.
    """

    output = widgets.Output()

    def _display_plot_section(section_title, df, plot_kwargs):
        if df is None or df.empty:
            return []

        section_widgets = [widgets.HTML(f"<b># {section_title}</b>")]
        figs = create_axiswise_plots2(df, **plot_kwargs)
        for key, fig in figs.items():
            fig_out = widgets.Output()
            with fig_out:
                print(f"{section_title}: {key}")
                display(fig)
            section_widgets.append(fig_out)
        return section_widgets

    def _pick_frame(plot_sources, *keys):
        for key in keys:
            if key in plot_sources and plot_sources[key] is not None:
                return plot_sources[key]
        return None

    def on_generate_clicked(b):
        with output:
            clear_output()  # Only once, at the start
            plot_sources = None
            if get_plot_dfs_func is not None:
                plot_sources = get_plot_dfs_func()

            if isinstance(plot_sources, dict):
                print("Generating plots from targeted query DataFrames...")
                try:
                    external_df = _pick_frame(plot_sources, "ET200_Data", "ET200", "external")
                    hf_df = _pick_frame(plot_sources, "HF_Data", "hf")
                    lf_df = _pick_frame(plot_sources, "LF_Data", "lf")

                    section_widgets = []
                    section_widgets.extend(_display_plot_section(
                        "External Sensor Values",
                        external_df,
                        dict(
                            color_column='Signal',
                            add_vlines=True,
                            normalize_method='minmax',
                            x_column='WCS_Y_mm',
                            color_palette=None,
                            max_display_points=20000,
                        ),
                    ))

                    if hf_df is not None and not hf_df.empty:
                        hf_df = filter_constant_HF_signals(hf_df)
                        if 'Signal' in hf_df.columns:
                            hf_df = hf_df[~hf_df['Signal'].isin(['ToolOrientation', 'WCSPosition'])]

                    section_widgets.extend(_display_plot_section(
                        "HF Data Figures",
                        hf_df,
                        dict(
                            color_column='Signal',
                            add_vlines=True,
                            normalize_method='minmax',
                            x_column='WCS_Y_mm',
                            color_palette=None,
                            max_display_points=20000,
                        ),
                    ))

                    section_widgets.extend(_display_plot_section(
                        "LF Data Figures",
                        lf_df,
                        dict(
                            color_column='Signal',
                            signal_color_mapping=False,
                            marker=True,
                            add_vlines=True,
                            normalize_method=None,
                            x_column='WCS_Y_mm',
                            max_display_points=20000,
                        ),
                    ))

                    if section_widgets:
                        display(widgets.VBox(section_widgets))
                    else:
                        print("Please provide at least one non-empty plot DataFrame.")
                except Exception as e:
                    print(f"Error during plot generation: {e}")
                return

            if get_df_func is not None:
                df = get_df_func()
            else:
                df = globals().get('selected_df', None)
            if df is not None and not df.empty:
                # Data processing if we use voila and want to keep the same plot generation code as in notebook_visualisation.ipynb
                df = filter_constant_HF_signals(df)
                df = df[~df['Signal'].isin(['ToolOrientation', 'WCSPosition'])]
                print("Generating plots for selected DataFrame...")
                try:
                    section_widgets = []

                    # External Sensor Values
                    section_widgets.extend(_display_plot_section(
                        "External Sensor Values",
                        df[df['DataOrigin'] == 'ET200_Data'] if 'DataOrigin' in df.columns else df,
                        dict(
                            color_column='Signal',
                            add_vlines=True,
                            normalize_method='minmax',
                            x_column='WCS_Y_mm',
                            color_palette=None,
                            max_display_points=20000,
                        ),
                    ))

                    # HF Data Figures
                    hf_df = df[df['DataOrigin'] == 'HF_Data'] if 'DataOrigin' in df.columns else df
                    section_widgets.extend(_display_plot_section(
                        "HF Data Figures",
                        hf_df,
                        dict(
                            color_column='Signal',
                            add_vlines=True,
                            normalize_method='minmax',
                            x_column='WCS_Y_mm',
                            color_palette=None,
                            max_display_points=20000,
                        ),
                    ))

                    # LF Data Figures
                    lf_df = df[df['DataOrigin'] == 'LF_Data'] if 'DataOrigin' in df.columns else df
                    section_widgets.extend(_display_plot_section(
                        "LF Data Figures",
                        lf_df,
                        dict(
                            color_column='Signal',
                            signal_color_mapping=False,
                            marker=True,
                            add_vlines=True,
                            normalize_method=None,
                            x_column='WCS_Y_mm',
                            max_display_points=20000,
                        ),
                    ))

                    if section_widgets:
                        display(widgets.VBox(section_widgets))
                    else:
                        print("Please provide at least one non-empty plot DataFrame.")
                except Exception as e:
                    print(f"Error during plot generation: {e}")
            else:
                print("Please select a DataFrame first.")

    generate_button = widgets.Button(description='Generate Plots')
    style_button(generate_button)
    generate_button.on_click(on_generate_clicked)
    display(widgets.VBox([generate_button, output]))
    
# Utilities, to keep the buttons matching across the Notebooks    
# Provenance: new in this thesis.
def style_button(button):
    button.style.button_color = '#990000'
    button.style.text_color = 'white'
    button.style.font_family = 'Arial'
    button.style.font_weight = 'bold'
    return button


# Flexible widget for selecting plate, slot, and arbitrary filters (addtion)
# Provenance: new in this thesis. Replaces the legacy `show_plate_widget` /
# `show_plate_selector` pair and adds the per-signal filter rows.
def show_plate_slot_filter_widget(default_plate=None, state=None):
    """
    Widget for selecting plate, slot, and adding arbitrary column filters.
    Uses provider for all data access. Assigns DataFrame to selected_df in global scope.
    """
    plates = provider.plates()
    if not plates:
        display(widgets.HTML("<b>No plates found in provider.</b>"))
        return

    # Get all columns except Platte and Nut
    schema_df = provider.schema()
    all_columns = list(schema_df["column_name"]) if "column_name" in schema_df.columns else list(schema_df.iloc[:,0])
    filterable_columns = [col for col in all_columns if col not in ("Platte", "Nut", "Value", "Time")] ## exclude columns that are not useful for filtering (adjust as needed)

    plate_dropdown = widgets.Dropdown(
        options=plates,
        value=default_plate if default_plate in plates else plates[0],
        description="Plate:",
        layout=widgets.Layout(width="200px")
    )

    def get_slot_options(plate):
        slots = provider.slots(plate)
        # Add 'All Slots' option mapped to None
        slot_options = [("All Slots", None)] + [(str(s), s) for s in slots] if slots else [("All Slots", None)]
        return slot_options

    slot_dropdown = widgets.Dropdown(
        options=get_slot_options(plate_dropdown.value),
        value=None,
        description="Slot:",
        layout=widgets.Layout(width="200px")
    )

    def update_slots(change):
        slot_dropdown.options = get_slot_options(change["new"])
        slot_dropdown.value = None

    plate_dropdown.observe(update_slots, names="value")

    # Container for dynamic filter rows
    filter_rows = []
    filter_box = widgets.VBox([])

    def add_filter_row(_=None):
        col_dropdown = widgets.Dropdown(
            options=filterable_columns,
            description="Column:",
            layout=widgets.Layout(width="180px")
        )
        val_dropdown = widgets.Dropdown(
            options=[],
            description="Value:",
            layout=widgets.Layout(width="180px")
        )
        remove_btn = widgets.Button(
            icon="times",
            description="Remove",
            layout=widgets.Layout(width="25px")
        )

        def update_values(change):
            plate = plate_dropdown.value
            slot = slot_dropdown.value
            col = change["new"]
            try:
                df = provider.df(plate, slot, fields=[col])
                vals = sorted(df[col].dropna().unique())
                val_dropdown.options = vals
            except Exception:
                val_dropdown.options = []

        col_dropdown.observe(update_values, names="value")
        # Trigger initial value load
        col_dropdown.value = filterable_columns[0] if filterable_columns else None

        def remove_row(_):
            filter_rows.remove(row)
            filter_box.children = [r["widget"] for r in filter_rows]

        remove_btn.on_click(remove_row)

        row = {
            "col_dropdown": col_dropdown,
            "val_dropdown": val_dropdown,
            "remove_btn": remove_btn,
            "widget": widgets.HBox([col_dropdown, val_dropdown, remove_btn])
        }
        filter_rows.append(row)
        filter_box.children = [r["widget"] for r in filter_rows]

    add_filter_btn = widgets.Button(
        description="Add Filter",
        icon="plus",
        layout=widgets.Layout(width="120px")
    )
    style_button(add_filter_btn)
    add_filter_btn.on_click(add_filter_row)

    output = widgets.Output()

    def on_confirm(_):
        output.clear_output()
        plate = plate_dropdown.value
        slot = slot_dropdown.value
        filters = {}
        for row in filter_rows:
            col = row["col_dropdown"].value
            val = row["val_dropdown"].value
            if col and val is not None:
                filters[col] = val

        all_filters = filters.copy()  # for printing later
        # Build kwargs for provider.df
        kwargs = {}
        if "DataOrigin" in filters:
            kwargs["data_origin"] = filters.pop("DataOrigin")
        if "Signal" in filters:
            kwargs["signals"] = [filters.pop("Signal")]

        # If slot is None ("All Slots"), load all slots for the plate and concatenate
        if slot is None:
            slot_list = provider.slots(plate)
            dfs = []
            for s in slot_list:
                df = provider.df(plate, s, **kwargs)
                if df is not None and not df.empty:
                    dfs.append(df)
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = None
        else:
            df = provider.df(plate, slot, **kwargs)

        # For other columns, filter in-memory
        for col, val in filters.items():
            if df is not None and not df.empty:
                df = df[df[col] == val]

        with output:
            if df is not None and not df.empty:
                builtins.selected_df = df
                if all_filters:
                    print("Filters applied:")
                    for k, v in all_filters.items():
                        print(f"  - {k}: {v}")
                else:
                    print("No additional filters applied.")
                if state is not None:
                    state.update({"plate": plate, "slot": slot, "df": df, "filters": filters})
            else:
                print("❌ No data found for the selected filters.")
            if slot is None:
                print(f"✅ DataFrame for Plate {plate}, ALL slots loaded as 'selected_df' (rows: {len(df) if df is not None else 0}, columns: {df.shape[1] if df is not None else 0})")
            else:
                print(f"✅ DataFrame for Plate {plate}, Slot {slot} loaded as 'selected_df' (rows: {len(df) if df is not None else 0}, columns: {df.shape[1] if df is not None else 0})")

    confirm_button = widgets.Button(
        description="Load DataFrame",
        button_style='success',
        layout=widgets.Layout(width="150px")
    )
    style_button(confirm_button)
    confirm_button.on_click(on_confirm)

    # Layout
    display(widgets.VBox([
        widgets.HBox([plate_dropdown, slot_dropdown, confirm_button]),
        widgets.HBox([add_filter_btn]),
        filter_box,
        output
    ]))

