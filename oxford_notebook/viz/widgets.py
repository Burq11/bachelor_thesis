# viz/widgets_digital_twin.py
from src import provider
import ipywidgets as widgets
from IPython.display import display, clear_output
from viz.visualizer import create_axiswise_plots2
import pandas as pd
import numpy as np
import builtins

# Widget for selecting plate and slot in one step. Assigns DataFrame to selected_df in global scope.
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
    

#########HEATMAP###############  IMPLEMENTATION BY OTHER STUDENT
def show_heatmap_widget(heatmap_state=None):
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
    
        platte = dropdown_platte.value
        bin_size_mm = slider_bin.value
    
        with output_platte:
            print(f"Creating heatmap for Plate {platte} with bin size {bin_size_mm} mm...")
    
            try:
                from src.data_processing import prepare_equal_bins_heatmap, analyze_platte, summarize_chatter_cases
                from viz.visualizer import plot_digital_twin_heatmap_gradient

                # Use provider to get all slots for the selected plate
                slots = provider.slots(platte)
                if not slots:
                    print(f"No slots found for Plate {platte}")
                    return
    
                # Build heatmap data 
                all_heatmap_data = []
                y_ranges = []
                for slot in slots:
                    df = provider.df(platte, slot)
                    if df is None or df.empty:
                        continue
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
                y_min_max = float(bin_max["Y_min"])
                y_max_max = float(bin_max["Y_max"])
        
                # Lists to collect X-axis oscillation values belonging to each bin
                vals_min = []   # raw vibrations in RMS_min bin
                vals_max = []   # raw vibrations in RMS_max bin
        
                # Loops over slots
                for slot in slots:
                    df = provider.df(platte, slot)
                    if df is None or df.empty:
                        continue
        
                    # Extract only oscilloscope X-channel data
                    df_sig = df[
                        (df["Axis"] == "X") &
                        (df["DataOrigin"] == "Oscilloscope")
                    ]
        
                    # OPTIONAL: remove extreme spikes if desired
                    # df_sig = df_sig[df_sig["Value"].between(-1.0, 1.0)]
        
                    # If this file corresponds to the RMS_min bin slot:
                    if slot == slot_min:
                        # Select oscilloscope samples exactly within this bin's Y-range
                        mask_min = (
                            (df_sig["WCS_Y_mm"] >= y_min_min) &
                            (df_sig["WCS_Y_mm"] <= y_max_min)
                        )
                        vals_min.extend(df_sig.loc[mask_min, "Value"].values)
                    # If this slot corresponds to the RMS_max bin slot:
                    if slot == slot_max:
                        mask_max = (
                            (df_sig["WCS_Y_mm"] >= y_min_max) &
                            (df_sig["WCS_Y_mm"] <= y_max_max)
                        )
                        vals_max.extend(df_sig.loc[mask_max, "Value"].values)
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
                df_summary, _ = analyze_platte(platte, summarize_chatter_cases, lambda x: None)
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

# Utility Voila: Button to trigger plot generation for the selected DataFrame.
def show_generate_plots_button(get_df_func=None):
    """
    Displays a button that generates plots for the selected DataFrame.
    get_df_func: Optional function that returns the DataFrame to plot (default: uses global 'selected_df').
    """

    output = widgets.Output()
    def on_generate_clicked(b):
        with output:
            clear_output()  # Only once, at the start
            if get_df_func is not None:
                df = get_df_func()
            else:
                df = globals().get('selected_df', None)
            if df is not None and not df.empty:
                # Data processing if we use voila and want to keep the same plot generation code as in notebook_visualisation.ipynb
                from src.data_processing import filter_constant_HF_signals
                df = filter_constant_HF_signals(df)
                df = df[~df['Signal'].isin(['ToolOrientation', 'WCSPosition'])]
                print("Generating plots for selected DataFrame...")
                try:
                    section_widgets = []

                    # External Sensor Values
                    ext_header = widgets.HTML("<b># External Sensor Values</b>")
                    ext_figs = create_axiswise_plots2(df, color_column='Signal', add_vlines=True, normalize_method='minmax', x_column='WCS_Y_mm', color_palette=None, max_display_points=20000, filter_value='Oscilloscope')
                    ext_outputs = []
                    for key, fig in ext_figs.items():
                        fig_out = widgets.Output()
                        with fig_out:
                            print(f"External: {key}")
                            display(fig)
                        ext_outputs.append(fig_out)
                    section_widgets.append(ext_header)
                    section_widgets.extend(ext_outputs)

                    # HF Data Figures
                    hf_header = widgets.HTML("<b># HF Data Figures</b>")
                    hf_figs = create_axiswise_plots2(df, color_column='Signal', add_vlines=True, normalize_method='minmax', x_column='WCS_Y_mm', color_palette=None, max_display_points=20000, filter_value='HF_Data')
                    hf_outputs = []
                    for key, fig in hf_figs.items():
                        fig_out = widgets.Output()
                        with fig_out:
                            print(f"HF: {key}")
                            display(fig)
                        hf_outputs.append(fig_out)
                    section_widgets.append(hf_header)
                    section_widgets.extend(hf_outputs)

                    # LF Data Figures
                    lf_header = widgets.HTML("<b># LF Data Figures</b>")
                    lf_figs = create_axiswise_plots2(df, color_column='Signal', signal_color_mapping=False, marker=True, add_vlines=True, normalize_method=None, x_column='WCS_Y_mm', max_display_points=20000, filter_value='LF_Data')
                    lf_outputs = []
                    for key, fig in lf_figs.items():
                        fig_out = widgets.Output()
                        with fig_out:
                            print(f"LF: {key}")
                            display(fig)
                        lf_outputs.append(fig_out)
                    section_widgets.append(lf_header)
                    section_widgets.extend(lf_outputs)

                    display(widgets.VBox(section_widgets))
                except Exception as e:
                    print(f"Error during plot generation: {e}")
            else:
                print("Please select a DataFrame first.")

    generate_button = widgets.Button(description='Generate Plots')
    style_button(generate_button)
    generate_button.on_click(on_generate_clicked)
    display(widgets.VBox([generate_button, output]))
    
# Utilities, to keep the buttons matching across the Notebooks    
def style_button(button):
    button.style.button_color = '#990000'
    button.style.text_color = 'white'
    button.style.font_family = 'Arial'
    button.style.font_weight = 'bold'
    return button


# Flexible widget for selecting plate, slot, and arbitrary filters (addtion)
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

    slot_dropdown = widgets.Dropdown(
        options=provider.slots(plate_dropdown.value),
        description="Slot:",
        layout=widgets.Layout(width="200px")
    )

    def update_slots(change):
        slots = provider.slots(change["new"])
        slot_dropdown.options = slots if slots else []
        if slots:
            slot_dropdown.value = slots[0]

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
        # For other columns, filter in-memory
        df = provider.df(plate, slot, **kwargs)
        for col, val in filters.items():
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
            
            print(f"✅ DataFrame for Plate {plate}, Slot {slot} loaded as 'selected_df' (rows: {len(df)}, columns: {df.shape[1]})")

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

