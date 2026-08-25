import itertools
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale
from scipy.cluster.hierarchy import linkage, leaves_list

from src.data_processing import filter_unique_gcodes
from viz import IWF_template
from viz.IWF_template import (
    register_templates,
    PTZ_colors,
    FraunhoferColors,
    IWF_Red_Fade,
    IWF_GreyBlue_fade,
    IWF_Brown_fade,
    IWF_Black_fade,
)

def add_gcode_vrects(fig, df_GCode, df, time_column, gcode_column, color1="white", color2=None):
    """
    Fügt vertikale Rechtecke (vrects) zu einem Plotly-Diagramm hinzu, basierend auf den ersten Vorkommen
    von GCode-Werten in einer angegebenen GCode-Spalte des DataFrames. Die Füllfarbe wechselt zwischen zwei Farben.

    Hinweis: Das Eingangs-DataFrame `df_GCode` muss bereits vorgefiltert sein, sodass es nur das erste Signal und die
    ersten Vorkommen jedes GCode-Wertes enthält. Dies kann durch die Funktion `filter_unique_gcodes` geschehen.

    Parameters
    ------
    fig : plotly.graph_objects.Figure
        Das Plotly-Diagramm, zu dem die vertikalen Rechtecke hinzugefügt werden sollen.
    df_GCode : pandas.DataFrame
        Der vorgefilterte DataFrame, der die GCode-Ereignisse enthält.
    df : pandas.DataFrame
        Der ursprüngliche DataFrame, der die Zeitangaben enthält.
    time_column : str
        Der Name der Spalte, die die Zeitangaben enthält (z.B. 'Duration_Seconds').
    gcode_column : str
        Der Name der Spalte, die die GCode-Werte enthält (z.B. 'GCode_Label').
    color1 : str, optional
        Die erste Farbe für die Füllung der Rechtecke (default ist "white").
    color2 : str or None, optional
        Die zweite Farbe für die Füllung der Rechtecke (default ist None).
    
    Returns
    -------
    None
        Die Funktion modifiziert das übergebene `fig`-Objekt direkt, indem vertikale Rechtecke hinzugefügt werden.

    Examples
    --------
    >>> fig = px.line(df, x='Duration_Seconds', y='Value', color='Signal')
    >>> add_gcode_vrects(fig, df_GCode, df, time_column='Duration_Seconds', gcode_column='GCode_Label', color1="green", color2="blue")
    >>> fig.show()
    """

    # Farben für den Wechsel festlegen
    colors = [color1, color2]

    # Hinzufügen von vertikalen Rechtecken zum Diagramm
    for i in range(len(df_GCode)):
        # Der Originalindex, um die korrekten Werte aus df_GCode zu holen
        x_val = df_GCode.iloc[i][time_column]
        an_val = df_GCode.iloc[i][gcode_column]
        
        # Überprüfen, ob es einen nächsten Wert gibt, um x1 zu definieren
        if i + 1 < len(df_GCode):
            x1_val = df_GCode.iloc[i + 1][time_column]
        else:
            x1_val = df[time_column].max()

        # Farbe wechseln
        fill_color = colors[i % 2]

        # Rechteck hinzufügen
        fig.add_vrect(x0=x_val, x1=x1_val, annotation_text=an_val, fillcolor=fill_color, opacity=0.25, line_width=1, annotation_position="top left")
        
def add_gcode_vlines(fig, df, time_column, gcode_column, line_color="grey"):
    """
    Fügt vertikale Linien (vlines) zu einem Plotly-Diagramm hinzu, basierend auf den ersten Vorkommen
    von GCode-Werten in einer angegebenen GCode-Spalte des DataFrames.

    Hinweis: Das Eingangs-DataFrame `df` muss bereits vorgefiltert sein, sodass es nur das erste Signal und die
    ersten Vorkommen jedes GCode-Wertes enthält. Dies kann durch die Funktion `filter_unique_gcodes` geschehen.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Das Plotly-Diagramm, zu dem die vertikalen Linien hinzugefügt werden sollen.
    df : pandas.DataFrame
        Der vorgefilterte DataFrame, der die Signale und GCode-Ereignisse enthält.
    time_column : str
        Der Name der Spalte, die die Zeitangaben enthält (z.B. 'Duration_Seconds').
    gcode_column : str
        Der Name der Spalte, die die GCode-Werte enthält (z.B. 'GCode_Label').
    line_color : str, optional
        Die Farbe der vertikalen Linien (default ist "grey").
    
    Returns
    -------
    None
        Die Funktion modifiziert das übergebene `fig`-Objekt direkt, indem vertikale Linien hinzugefügt werden.

    Examples
    --------
    >>> import plotly.express as px
    >>> df = px.data.iris()  # Beispiel-DatenFrame
    >>> fig = px.line(df, x='sepal_width', y='sepal_length')
    >>> add_gcode_vlines(fig, df, time_column='sepal_width', gcode_column='species', line_color="blue")
    >>> fig.show()
    """
    
    # Vektorisierte Methode zum Hinzufügen von vertikalen Linien
    x_vals = df[time_column].values
    an_vals = df[gcode_column].values
    
    for x_val, an_val in zip(x_vals, an_vals):
        fig.add_vline(x=x_val, annotation_text=an_val, line_color=line_color)


# NOTE: currently unused — no callers found anywhere in the codebase (only its own docstring
# examples reference it). Kept pending review.
def create_plot_with_gcode_annotations(df,
                                       filter_column='DataOrigin', 
                                       filter_value='HF_Data', 
                                       axis_column='Axis', 
                                       x_column='Duration_Seconds', 
                                       y_column='Value', 
                                       color_column='Signal', 
                                       hoverdict=None, 
                                       max_display_points=10000,
                                       normalize_method='zscore',  # oder 'minmax' oder None
                                       add_vrects=False, 
                                       add_vlines=False, 
                                       vrect_color1="white", 
                                       vrect_color2=None,
                                       axis_sort_order=['X', 'Y', 'Z', 'SP', 'A', 'B', '', np.nan],
                                       trace_opacity=0.6):
    """
    Erstellt Plotly-Diagramme in einer bestimmten Reihenfolge basierend auf den Werten im `axis_column`.
    Optional können GCode-Annotationen (z. B. Rechtecke oder Linien) sowie eine Normalisierung der Signale
    (z. B. Min-Max oder z-Score) durchgeführt werden. Zusätzlich wird automatisch ein intelligentes 
    Downsampling vorgenommen, um die Darstellbarkeit zu verbessern.

    Parameters
    ----------
    df : pandas.DataFrame
        Der Eingabe-DataFrame mit den Daten für das Diagramm.
    filter_column : str, optional
        Der Name der Spalte, die zum Filtern des DataFrames verwendet wird (default ist 'DataOrigin').
    filter_value : any, optional
        Der Wert, nach dem der DataFrame gefiltert wird (default ist 'HF_Data').
    axis_column : str, optional
        Der Name der Spalte, die verwendet wird, um den DataFrame in unterschiedliche Achsen aufzuteilen (default ist 'Axis').
    x_column : str, optional
        Der Name der Spalte, die für die x-Achse verwendet wird (default ist 'Duration_Seconds').
    y_column : str, optional
        Der Name der Spalte, die für die y-Achse verwendet wird (default ist 'Value').
    color_column : str, optional
        Der Name der Spalte, die für die Färbung der Linien verwendet wird (default ist 'Signal').
    hoverdict : dict, optional
        Ein Dictionary, das zusätzliche Spalten für den Hover-Text angibt (default ist None).
    max_display_points : int, optional
        Die maximale Anzahl an Punkten, die pro Achse nach dem Downsampling dargestellt werden (default ist 10.000).
    normalize_method : str or None, optional
        Die Methode zur Normalisierung der `y_column`-Werte pro Signal. 
        Erlaubte Werte sind 'zscore' (Standard), 'minmax' oder None für keine Normalisierung.
    add_vrects : bool, optional
        Ob vertikale Rechtecke basierend auf GCode-Ereignissen hinzugefügt werden sollen (default ist False).
    add_vlines : bool, optional
        Ob vertikale Linien basierend auf GCode-Ereignissen hinzugefügt werden sollen (default ist False).
    vrect_color1 : str, optional
        Die erste Farbe für die Rechteckfüllung (default ist "white").
    vrect_color2 : str or None, optional
        Die zweite Farbe für die Rechteckfüllung (optional für abwechselnde Farbtöne).
    axis_sort_order : list, optional
        Eine Liste von Werten, die die Reihenfolge bestimmt, in der die Plots für die Achsenwerte erstellt werden.
    trace_opacity : float, optional
        Die Opazität der Linien-Traces im Diagramm (default ist 0.6).

    Returns
    -------
    None
        Die Funktion zeigt die erzeugten Plotly-Diagramme direkt in der Notebook-Ausgabe an.

    Examples
    --------
    >>> create_plot_with_gcode_annotations(df, axis_sort_order=['X', 'Y', 'Z'])
    >>> create_plot_with_gcode_annotations(df, normalize_method='minmax', add_vrects=True)
    >>> create_plot_with_gcode_annotations(df, normalize_method=None, max_display_points=5000)
    """

    if hoverdict is None:
        hoverdict = {
            'Unit': True,
            'Signal_Label': True,
            'Groupname': True,
            'GCode_Label': True,
            'Cycle': True
        }

    df_filtered = df.loc[df[filter_column] == filter_value].copy()
    df_filtered[color_column] = df_filtered[color_column].astype(str)

    # Normierung abhängig vom Parameter
    if normalize_method == 'zscore':
        df_filtered[y_column] = (
            df_filtered.groupby(color_column)[y_column]
            .transform(lambda x: (x - x.mean()) / x.std(ddof=0))
        )
    elif normalize_method == 'minmax':
        df_filtered[y_column] = (
            df_filtered.groupby(color_column)[y_column]
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        )
    elif normalize_method is not None:
        raise ValueError("normalize_method must be 'zscore', 'minmax', or None")

    df_GCode = filter_unique_gcodes(df)
    available_axis_values = df_filtered[axis_column].unique()
    axis_values = [axis for axis in axis_sort_order if axis in available_axis_values]

    def downsample_dataframe(df_sub, max_points=10000):
        if len(df_sub) <= max_points:
            return df_sub
        else:
            idx = np.linspace(0, len(df_sub) - 1, max_points).astype(int)
            return df_sub.iloc[idx]
        
    # figures = []

    for axis_value in axis_values:
        _df = df_filtered.loc[df_filtered[axis_column] == axis_value]
        _df = downsample_dataframe(_df, max_points=max_display_points)

        if _df.empty:
            continue

        fig = px.line(
            _df, 
            x=x_column, 
            y=y_column, 
            color=color_column, 
            hover_data=hoverdict, 
            title=f"{axis_column}: {axis_value}", 
            markers=True
        )

        if add_vrects:
            add_gcode_vrects(fig, df_GCode, _df, time_column=x_column, gcode_column='GCode_Label',
                             color1=vrect_color1, color2=vrect_color2)

        if add_vlines:
            add_gcode_vlines(fig, df_GCode, time_column=x_column, gcode_column='GCode_Label')

        fig.update_traces(opacity=trace_opacity)
#         figures.append(fig)
                
# #Am Ende: alle anzeigen
#         r fig in figures:
#           display(fig)
#         import ipywidgets as widgets
#         from IPython.display import display
#         out = widgets.Output()
#         with out:
#             display(fig)
#         display(out)

## we are not using this anywhere
# def plot_digital_twin_segmented(df_summary, nut_width=5.0):
#     """
#     Visualisiert eine Fräsplatte basierend auf zusammengefassten Nutdaten
#     mit Chatter-Markierung (stabil/instabil) als digitale Zwilling-Grafik.

#     Für jede Nut wird anhand des 'Chatter'-Labels und der Y_max-Werte ein
#     rechteckiger Bereich gezeichnet:
#     - Chatter = 0 → stabiler Bereich
#     - Chatter = 1 → instabiler Bereich
#     - beide → segmentiert in oberen/unteren Bereich

#     Parameter
#     ----------
#     df_summary : pd.DataFrame
#         Aggregierter DataFrame mit Spalten:
#         ['Chatter', 'Y_max', 'Drehzahl', 'Werkzeugradius', 'X_Position_Nut', 'Nut_ID', ...]
#     nut_width : float, optional
#         Breite der Nuten für die Darstellung (nicht verwendet, x-Position wird aus Radius berechnet)

#     Returns
#     -------
#     fig : go.Figure
#         Interaktive Plotly-Grafik der Fräsplatte mit Chattersegmentierung.
#     """
#     fig = go.Figure()

#     # Grundplatte
#     fig.add_shape(type="rect", x0=0, x1=245, y0=0, y1=245,
#                   fillcolor=IWF_template.FraunhoferColors[4], line=dict(color="black"))

#     for nut_id in sorted(df_summary["Nut_ID"].unique()):
#         gruppe = df_summary[df_summary["Nut_ID"] == nut_id].sort_values("Chatter")
#         x_pos = gruppe["X_Position_Nut"].iloc[0]
#         radius = gruppe["Werkzeugradius"].iloc[0]
#         rpm = gruppe["Drehzahl"].iloc[0]
#         x0 = x_pos - radius
#         x1 = x_pos + radius

#         if len(gruppe) == 2:
#             y_split = gruppe["Y_max"].iloc[0]
#             y_max = gruppe["Y_max"].iloc[1]
#             fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=y_split,
#                           fillcolor=IWF_template.FraunhoferColors[0], line=dict(color="black"), opacity=1)
#             fig.add_shape(type="rect", x0=x0, x1=x1, y0=y_split, y1=y_max,
#                           fillcolor=IWF_template.iwfColors_without_white[3], line=dict(color="black"), opacity=1)
#         else:
#             y_max = gruppe["Y_max"].iloc[0]
#             color = IWF_template.iwfColors_without_white[1] if gruppe["Chatter"].iloc[0] == 1 else IWF_template.FraunhoferColors[0]
#             fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=y_max,
#                           fillcolor=color, line=dict(color="black"), opacity=1)

#         # Nutnummer (oben)
#         fig.add_annotation(
#             x=x_pos,
#             y=0,
#             text=f"N{int(nut_id)}",
#             showarrow=False,
#             font=dict(size=10, color='black'),
#             textangle=0,
#             yanchor='top'
#         )

#         # Drehzahl (unten)
#         fig.add_annotation(
#             x=x_pos,
#             y=1,
#             text=f"{int(rpm)} rpm",
#             showarrow=False,
#             font=dict(size=10, color='black'),
#             textangle=-90,
#             yanchor='bottom'
#         )

#     fig.update_layout(
#         title="Digital Twin der Fräsplatte (Chatter-Visualisierung)",
#         xaxis_title="X [mm]",
#         yaxis_title="Y [mm]",
#         xaxis=dict(range=[0, 250]),
#         yaxis=dict(range=[-10, 270]),
#         width=600,
#         height=600,
#         plot_bgcolor="white"
#     )
#     fig.update_yaxes(scaleanchor="x", scaleratio=1)
#     return fig

def downsample_df(df, max_points: int = 10000):
    """
    Reduziert die Anzahl der Datenpunkte gleichmäßig über alle Zeilen hinweg.

    Diese Funktion eignet sich besonders zur visuellen Darstellung von Zeitreihendaten,
    bei denen eine zu hohe Auflösung Performanceprobleme verursacht.

    Parameter
    ----------
    df : pandas.DataFrame
        Eingabedaten mit beliebiger Länge.
    max_points : int, optional
        Maximale Anzahl der zurückgegebenen Zeilen. Standard: 10000.

    Returns
    -------
    pandas.DataFrame
        Gleichmäßig reduzierter DataFrame.
    """
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx]

def get_signal_to_color_map(df, signal_column, group_column, color_palette):
    """
    Erstellt ein Mapping von Signalen zu Farben auf Basis ihrer Gruppenzugehörigkeit.

    Nutzt eine zugeordnete Farbpalette, um eindeutige Farben je Gruppe zu vergeben.
    Diese Gruppenfarben werden anschließend den Signalen zugeordnet.

    Parameter
    ----------
    df : pandas.DataFrame
        DataFrame mit mindestens signal_column und group_column.
    signal_column : str
        Name der Spalte mit Signalbezeichnungen.
    group_column : str
        Name der Gruppierungsspalte, z. B. 'Groupname'.
    color_palette : list[str]
        Liste von Farbwerten, die zyklisch den Gruppen zugewiesen werden.

    Returns
    -------
    dict[str, str]
        Dictionary: Signalname → Farbwert.
    """
    unique_groups = df[group_column].dropna().unique()
    group_to_color = {
        group: color for group, color in zip(sorted(unique_groups), itertools.cycle(color_palette))
    }
    signal_to_color = (
        df.dropna(subset=[signal_column, group_column])
        .drop_duplicates(subset=[signal_column])
        .set_index(signal_column)[group_column]
        .map(group_to_color)
        .to_dict()
    )
    return signal_to_color

def normalize_column(df, column, groupby_col, method='zscore'):
    """
    Normalisiert eine numerische Spalte gruppenweise nach dem gewählten Verfahren.

    Unterstützt z-Score-Normalisierung und Min-Max-Skalierung.
    Wird keine Normalisierung gewünscht, bleibt die Spalte unverändert.

    Parameter
    ----------
    df : pandas.DataFrame
        Eingabedaten.
    column : str
        Name der Spalte, die normalisiert werden soll.
    groupby_col : str
        Gruppierungsspalte, nach der die Normalisierung durchgeführt wird.
    method : {'zscore', 'minmax', None}
        Verfahren zur Normalisierung. Standard: 'zscore'.

    Returns
    -------
    pandas.Series
        Normalisierte Spalte (als neue Series).
    """
    if method == 'zscore':
        return df.groupby(groupby_col)[column].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    elif method == 'minmax':
        return df.groupby(groupby_col)[column].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    elif method is None:
        return df[column]
    else:
        raise ValueError(f"Unbekannte Normalisierungsmethode: {method}")


# NOTE: currently unused — no callers found. The variant create_axiswise_plots2 is the one
# actually used (widgets.py, notebooks). Kept pending review.
def create_axiswise_plots(
    df,
    axis_column='Axis',
    filter_column='DataOrigin',
    filter_value='HF_Data',
    signal_column='Signal',
    group_column='Groupname',
    x_column='Duration_Seconds',
    y_column='Value',
    color_column=None,
    hoverdict=None,
    normalize_method='zscore',
    max_display_points=10000,
    sort_order=['X', 'Y', 'Z', 'SP', 'A', 'B', ''],
    color_palette=None
):
    """
    Erstellt gruppierte Linienplots je Achse aus CNC-Daten oder Messzeitreihen.

    Farbpalette wird automatisch aus dem aktiven Plotly-Template übernommen,
    sofern keine benutzerdefinierte übergeben wird.

    Parameter
    ----------
    df : pandas.DataFrame
        Eingabedaten, z. B. CNC-Zeitreihen mit Spalten wie Signal, Axis, Value.
    axis_column : str
        Spalte zur Gruppierung nach Achsen (z. B. 'Axis').
    filter_column : str
        Spalte zur Filterung der Datenquelle (z. B. 'DataOrigin').
    filter_value : str
        Wert zur Filterung der Datenquelle (z. B. 'HF_Data').
    signal_column : str
        Spalte mit Signalnamen.
    group_column : str
        Gruppierungsmerkmal zur Farbcodierung (z. B. 'Groupname').
    x_column : str
        Spalte für die x-Achse (z. B. 'Duration_Seconds').
    y_column : str
        Spalte für die y-Achse (z. B. 'Value').
    color_column : str or None    
        Wenn gesetzt, wird eine Farbcodierung je Signal angewendet.
    hoverdict : dict, optional
        Zusätzliche Felder für Hoveranzeigen.
    normalize_method : {'zscore', 'minmax', None}
        Verfahren zur Normalisierung der y-Achse. Standard: 'zscore'.
    max_display_points : int
        Maximale Anzahl Punkte pro Kurve (Downsampling).
    sort_order : list[str]
        Reihenfolge der Achsen für die Ausgabe.
    color_palette : list[str] or None
        Optional: eigene Farbpalette. Wenn None, wird die Palette aus dem aktiven Plotly-Template verwendet.

    Returns
    -------
    dict[str, plotly.graph_objects.Figure]
        Dictionary mit Achsennamen als Keys und Plotly-Figuren als Werten.
    """
    if hoverdict is None:
        hoverdict = {}

    # Filterung nach Datenquelle
    df_filtered = df.loc[df[filter_column] == filter_value].copy()

    # Farbpalette aus Template laden, wenn nicht angegeben
    if color_palette is None:
        template_name = pio.templates.default
        color_palette = pio.templates[template_name].layout.colorway

    # Signal zu Farbe zuordnen
    signal_to_color = get_signal_to_color_map(
        df_filtered, signal_column, group_column, color_palette
    )

    # Normalisierung der y-Werte
    df_filtered[y_column] = normalize_column(
        df_filtered, y_column, groupby_col=signal_column, method=normalize_method
    )

    # Sortierung der Achsen nach Vorgabe
    available_axes = df_filtered[axis_column].dropna().unique()
    axis_values = [a for a in sort_order if a in available_axes]

    figures = {}

    for axis in axis_values:
        df_axis = df_filtered[df_filtered[axis_column] == axis].copy()
        if df_axis.empty:
            continue

        # Downsampling je Signal
        df_axis = df_axis.groupby(signal_column, group_keys=False).apply(
            lambda d: downsample_df(d, max_points=max_display_points),
        )

        # Plot erstellen
        fig = px.line(
            df_axis,
            x=x_column,
            y=y_column,
            color=signal_column,
            hover_data=hoverdict,
            title=f"Achse: {axis}",
            color_discrete_map=signal_to_color
        )
        fig.update_layout(showlegend=True)

        figures[axis] = fig

    return figures

def create_axiswise_plots2(
    df,
    axis_column='Axis',
    signal_column='Signal',
    group_column='Groupname',
    x_column='Duration_Seconds',
    y_column='Value',
    color_column=None,
    signal_color_mapping=True,
    hoverdict=None,
    marker=False,
    normalize_method='zscore',
    max_display_points=10000,
    sort_order=['X', 'Y', 'Z', 'SP', 'A', 'B', ''],
    color_palette=None,
    trace_opacity=0.6,
    add_vrects=False,
    add_vlines=False,
    gcode_column='GCode_Label',
    gcode_time_column='Time',
    vrect_color1="white",
    vrect_color2=None
):
    """
    Erstellt gruppierte Plotly-Linienplots je Achse inkl. optionaler GCode-Annotationen.

    Die Farbpalette wird aus dem aktiven Plotly-Template oder optional übergeben.
    Ereignisse wie NC-Blöcke oder Taktpunkte können als vertikale Rechtecke oder Linien dargestellt werden.

    Returns
    -------
    dict[str, plotly.graph_objects.Figure]
        Dictionary mit Achsennamen als Keys und Plotly-Figuren als Werten.
    """
    # Assume the incoming DataFrame is already filtered by `DataOrigin` upstream.
    # Work on a copy to avoid mutating the caller's DataFrame.
    df_filtered = df.copy()

    # Guard: an empty DataFrame carries no columns, so every column access below
    # would raise. Upstream (provider.axiswise_plot_df) already explains *why* it
    # is empty - here we just return no figures instead of crashing.
    if df_filtered.empty:
        warnings.warn(
            "create_axiswise_plots2: empty Data Frame, can't generate a plot.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    # Guard: required columns must exist in the provided (pre-filtered) DataFrame.
    required_cols = [signal_column, x_column, y_column, axis_column]
    missing = [c for c in required_cols if c not in df_filtered.columns]
    if missing:
        raise ValueError(f"create_axiswise_plots2: missing required columns in df (expected): {missing}")

    df_filtered[axis_column] = df_filtered[axis_column].fillna("not axis specific")

    if hoverdict is None:
        hoverdict = {
            'Unit': True,
            'Signal_Label': True,
            'Groupname': True,
            'GCode_Label': True,
            'Cycle': True
        }

    df_filtered[signal_column] = df_filtered[signal_column].astype(str)

    if color_palette is None:
        template_name = pio.templates.default
        color_palette = pio.templates[template_name].layout.colorway
    if signal_color_mapping == True:
        signal_to_color = get_signal_to_color_map(
            df_filtered, signal_column, group_column, color_palette
        )
    else:
        signal_to_color = None

    df_filtered[y_column] = normalize_column(
        df_filtered, y_column, groupby_col=signal_column, method=normalize_method
    )

    # Use the filtered DataFrame when extracting unique GCode events so annotations align with plotted data.
    df_gcode = filter_unique_gcodes(df_filtered, signal_column=signal_column, gcode_column=gcode_column, sort_column=gcode_time_column)
    available_axes = df_filtered[axis_column].unique()
    axis_values = sorted(
        available_axes, key=lambda x: sort_order.index(x) if x in sort_order else len(sort_order)
    )

    figures = {}

    for axis in axis_values:
        df_axis = df_filtered[df_filtered[axis_column] == axis].copy()
        if df_axis.empty:
            continue

        # Faster deterministic downsampling: process groups and concat
        groups = []
        for _, grp in df_axis.groupby(signal_column, sort=False):
            groups.append(downsample_df(grp, max_points=max_display_points))
        if groups:
            df_axis = pd.concat(groups)
        else:
            df_axis = df_axis.iloc[0:0]

        fig = px.line(
            df_axis,
            x=x_column,
            y=y_column,
            color=signal_column,
            hover_data=hoverdict,
            title=f"Achse: {axis}",
            color_discrete_map=signal_to_color,
            markers=marker
        )
        fig.update_traces(opacity=trace_opacity)

        if add_vrects:
            add_gcode_vrects(
                fig, df_gcode, df_axis,
                time_column=x_column,
                gcode_column=gcode_column,
                color1=vrect_color1,
                color2=vrect_color2
            )

        if add_vlines:
            add_gcode_vlines(
                fig, df_gcode,
                time_column=x_column,
                gcode_column=gcode_column
            )

        figures[axis] = fig

    return figures
    
#########HEATMAP###############
def plot_digital_twin_heatmap_gradient(
    df_heatmap,
    df_summary=None,
    slot_column='Nut',
    y_column='Y_bin_center',
    intensity_column='RMS_normalized_global',
    plate_width=245,
    plate_height=245,
):
    """
    IWF-style Digital Twin Heatmap with:
    - red fade heatmap for vibration intensity,
    - MRR Qw iso-lines based on experimental parameters,
    - additional right-hand y-axis showing MRR Qw [mm³/min].

    Experimental parameters:
        a_e   = 10 mm (tool ED_M_12)
        f_rev = 0.18 mm/rev (G95)
        a_p   = 2.0 ... 10.56 mm (tilted table)
        Qw    = a_e * f_rev * a_p * n = 1.8 * a_p * n
    """

    # Empty fallbacks
    if df_heatmap.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No heatmap data available",
            x=plate_width / 2,
            y=plate_height / 2,
            showarrow=False,
            font=dict(size=20),
        )
        fig.update_layout(
            title="Digital Twin Heatmap - No Data",
            xaxis=dict(range=[0, plate_width]),
            yaxis=dict(range=[0, plate_height]),
            paper_bgcolor="black",
        )
        return fig

    fig = go.Figure()

    # Base plate (over grid; under Qw lines) 
    base_color = FraunhoferColors[4]
    border_color = PTZ_colors[0]
    fig.add_shape(
        type="rect",
        x0=0,
        x1=plate_width,
        y0=0,
        y1=plate_height,
        fillcolor=base_color,
        line=dict(color=border_color, width=2),
        opacity=1.0,
        layer="above",
    )

    # Color gradient for vibration intensity 
    start_hex, end_hex = IWF_Red_Fade

    def hex_to_rgb(h):
        return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])

    start_rgb, end_rgb = hex_to_rgb(start_hex), hex_to_rgb(end_hex)

    def get_color(v):
        v = np.clip(v, 0, 1)
        rgb = (1 - v) * start_rgb + v * end_rgb
        return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"

    # Slot positioning: use DB-provided Nut_ID -> X_Position_Nut mapping.
    slots = sorted(df_heatmap[slot_column].unique())
    slot_radius = 2.5

    slot_positions = {}
    if df_summary is not None and "X_Position_Nut" in df_summary.columns and "Nut_ID" in df_summary.columns:
        try:
            slot_positions = df_summary.set_index("Nut_ID")["X_Position_Nut"].to_dict()
        except Exception:
            slot_positions = {}

        if "Werkzeugradius" in df_summary.columns and not df_summary["Werkzeugradius"].dropna().empty:
            try:
                slot_radius = float(df_summary["Werkzeugradius"].dropna().iloc[0])
            except Exception:
                pass

    # Draw heatmap strips for each slot (staged: build shapes/traces/annotations lists first)
    total_slot_shapes = 0
    total_slot_traces = 0

    staged_shapes = []
    staged_traces = []
    staged_annotations = []

    for slot_id in slots:
        slot_data = df_heatmap[df_heatmap[slot_column] == slot_id].copy()
        if slot_data.empty:
            continue

        # Expect `slot_id` to correspond to `Nut_ID` (integer-like). Use DB mapping.
        try:
            key = int(slot_id)
        except Exception:
            key = slot_id

        x_pos = slot_positions.get(key)
        if x_pos is None:
            # fallback to simple spacing if DB position missing
            try:
                x_pos = 10 + float(slot_id) * 15
            except Exception:
                x_pos = 10
        x0, x1 = x_pos - slot_radius, x_pos + slot_radius
        slot_data = slot_data.sort_values(y_column)

        # Ensure Y starts at 0 for each slot
        if "Y_min" in slot_data.columns:
            slot_data.iloc[0, slot_data.columns.get_loc("Y_min")] = 0
        else:
            slot_data.iloc[0, slot_data.columns.get_loc(y_column)] = 0

        y_max = (
            slot_data["Y_max"].max()
            if "Y_max" in slot_data
            else slot_data[y_column].max()
        )

        # Colored segments (build shapes, add invisible scatter traces for hover)
        added_shapes = 0
        added_traces = 0
        for _, row in slot_data.iterrows():
            y0 = row.get("Y_min", 0)
            y1 = row.get("Y_max", row[y_column])
            color = get_color(row.get(intensity_column, 0))

            staged_shapes.append(
                dict(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor=color,
                    line=dict(width=0),
                    opacity=1.0,
                    layer="above",
                )
            )
            added_shapes += 1

            # invisible scatter for hover
            staged_traces.append(
                go.Scatter(
                    x=[(x0 + x1) / 2],
                    y=[(y0 + y1) / 2],
                    mode="markers",
                    marker=dict(size=8, opacity=0),
                    hovertemplate=(
                        f"<b>Slot N{slot_id}</b><br>"
                        f"X: {x_pos:.1f} mm<br>"
                        f"Y: {y0:.1f}–{y1:.1f} mm<br>"
                        f"Vibration intensity: {row.get(intensity_column, 0):.3f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            added_traces += 1

        # Slot outline (above)
        staged_shapes.append(
            dict(
                type="rect",
                x0=x0,
                x1=x1,
                y0=0,
                y1=y_max,
                fillcolor="rgba(0,0,0,0)",
                line=dict(color=border_color, width=2),
                layer="above",
            )
        )
        added_shapes += 1

        # Slot label N#
        staged_annotations.append(
            dict(
                x=x_pos,
                y=-6,
                text=f"N{int(slot_id)}",
                showarrow=False,
                font=dict(size=10, color=border_color),
                yanchor="top",
            )
        )

        # Spindle speed label (if available)
        if df_summary is not None and "Drehzahl" in df_summary.columns:
            try:
                rpm = int(
                    df_summary[df_summary["Nut_ID"] == slot_id]["Drehzahl"].iloc[0]
                )
                staged_annotations.append(
                    dict(
                        x=x_pos,
                        y=1,
                        text=f"{rpm} rpm",
                        showarrow=False,
                        font=dict(size=8, color=border_color),
                        textangle=-90,
                        yanchor="bottom",
                    )
                )
            except Exception:
                pass

        total_slot_shapes += added_shapes
        total_slot_traces += added_traces
    # Attach staged shapes/traces/annotations in bulk
    existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    existing_shapes.extend(staged_shapes)
    fig.update_layout(shapes=existing_shapes)

    if staged_traces:
        fig.add_traces(staged_traces)

    existing_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    existing_annotations.extend(staged_annotations)
    fig.update_layout(annotations=existing_annotations)

    # MRR Qw iso-lines (using experimental parameters) 
    qw_levels = []
    y_ticks_axis2 = []
    tick_labels_axis2 = []

    if df_summary is not None and "Drehzahl" in df_summary.columns:
        a_e = 10.0      # mm
        f_rev = 0.18    # mm/rev
        k = a_e * f_rev  # 1.8

        ap_start = 2.0
        ap_end = 10.56

        slot_rpm = {}
        for slot_id in slots:
            try:
                rpm_val = float(
                    df_summary.loc[df_summary["Nut_ID"] == slot_id, "Drehzahl"].iloc[0]
                )
                slot_rpm[slot_id] = rpm_val
            except Exception:
                continue

        if slot_rpm:
            max_rpm = max(slot_rpm.values())
            qw_max = k * ap_end * max_rpm  # mm³/min

            # Qw levels every 50,000 mm³/min 
            level_max = int(np.floor(qw_max / 50000.0)) * 50000
            if level_max >= 50000:
                qw_levels = list(range(50000, level_max + 1, 50000))
            else:
                qw_levels = []

            # Continuous grey–blue colors for Qw levels
            # Define colorscale (like the red fade, but grey–blue)
            IWF_GreyBlue_fade_scale = [
                [0.0, "#dfe7ec"],
                [0.5, "#9fb6c4"],
                [1.0, "#3a515f"],
            ]

            if len(qw_levels) <= 1:
                positions = [0.5]
            else:
                positions = [
                    i / (len(qw_levels) - 1) for i in range(len(qw_levels))
                ]

            qw_colors = sample_colorscale(IWF_GreyBlue_fade_scale, positions)

            # For each Qw level, compute targets vectorized and coalesce contiguous points
            qw_shape_count = 0
            qw_staged_shapes = []

            # Build arrays for slots (stable ordering)
            slot_ids = np.array(slots)
            x_array = np.empty(len(slot_ids), dtype=float)
            rpm_array = np.full(len(slot_ids), np.nan, dtype=float)
            for i, sid in enumerate(slot_ids):
                # rpm
                try:
                    rpm_array[i] = float(slot_rpm.get(sid, np.nan))
                except Exception:
                    rpm_array[i] = np.nan
                # x positions (try int key then fallback)
                try:
                    key = int(sid)
                except Exception:
                    key = sid
                try:
                    x_array[i] = float(slot_positions.get(key, 10 + (float(sid) if isinstance(sid, (int, float)) or str(sid).replace('.','',1).isdigit() else 0) * 15))
                except Exception:
                    x_array[i] = float(slot_positions.get(key, 10))

            # Avoid divide-by-zero / nan propagation
            valid_rpm_mask = np.isfinite(rpm_array) & (rpm_array > 0)

            for qw_level, col in zip(qw_levels, qw_colors):
                # vectorized ap_target
                with np.errstate(divide='ignore', invalid='ignore'):
                    ap_target = qw_level / (k * rpm_array)

                mask = valid_rpm_mask & (ap_target >= ap_start) & (ap_target <= ap_end)
                if not mask.any():
                    continue

                y_target = plate_height * (ap_target - ap_start) / (ap_end - ap_start)

                # Find contiguous runs of True in mask and build SVG path per run
                idx = np.where(mask)[0]
                # identify breaks where consecutive indices are not sequential
                breaks = np.where(np.diff(idx) != 1)[0]
                starts = np.concatenate(([idx[0]], idx[breaks + 1]))
                ends = np.concatenate((idx[breaks], [idx[-1]]))

                for s, e in zip(starts, ends):
                    # build path string: Move to first point, then line to subsequent
                    path_pts = [f"M {x_array[s]:.2f},{y_target[s]:.2f}"]
                    if e > s:
                        for j in range(s + 1, e + 1):
                            path_pts.append(f"L {x_array[j]:.2f},{y_target[j]:.2f}")
                    path_str = " ".join(path_pts)

                    qw_staged_shapes.append(
                        dict(
                            type="path",
                            path=path_str,
                            line=dict(color=col, width=2, dash="dot"),
                            layer="above",
                        )
                    )
                    qw_shape_count += 1

            # attach Qw shapes in bulk (extend existing layout shapes)
            if qw_staged_shapes:
                existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
                existing_shapes.extend(qw_staged_shapes)
                fig.update_layout(shapes=existing_shapes)

            # tick positions for right-hand MRR axis (use max rpm)
            for qw_level in qw_levels:
                ap_axis = qw_level / (k * max_rpm)
                if ap_start <= ap_axis <= ap_end:
                    y_axis = plate_height * (ap_axis - ap_start) / (ap_end - ap_start)
                    y_ticks_axis2.append(y_axis)
                    tick_labels_axis2.append(f"{qw_level:,.0f}")

    # Dummy trace for yaxis2 (needed so y2 exists)
    if y_ticks_axis2:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                xaxis='x',
                yaxis='y2',
                showlegend=False,
                hoverinfo='skip',
            )
        )

    # Colorbar for vibration intensity (further right, smaller, lower)
    colorbar_trace = go.Heatmap(
        z=[[0, 1]],
        colorscale=IWF_Red_Fade,
        showscale=True,
        colorbar=dict(
            title=dict(
                text="<b>Vibration<br>Intensity</b>",
                side="right",
                font=dict(size=11, color="black"),
            ),
            tickvals=[0, 0.5, 1],
            ticktext=["Low 0", "0.5", "High 1"],
            tickfont=dict(size=9, color="black"),
            thickness=14,
            len=0.60,       # shorter
            x=1.22,         # moved further right
            y=0.44,         # slightly lower
            xanchor="left",
            bgcolor="rgba(255,255,255,1)",
            outlinewidth=1.2,
            outlinecolor="black",
        ),
        hoverinfo="none",
        opacity=0,
    )
    fig.add_trace(colorbar_trace)

    # Layout / main axes
    fig.update_layout(
        title=dict(
            text="Digital Twin der Fräsplatte (Chatter-Visualisierung Heatmap)",
            font=dict(size=15, color="white"),
            x=0.5, xanchor="center", yanchor="top", y=0.96,
        ),
        width=640,
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="black",
        template="IWF_template",
        margin=dict(l=80, r=150, t=80, b=100),  # space for axis titles
        legend=dict(
            x=0.66,
            y=0.96,
            font=dict(size=8.5, color="white"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title=dict(text="X [mm]", font=dict(size=12, color="white")),
            title_standoff=40,
            range=[-15, 265],
            tick0=0,
            dtick=50,                 # grid every 50 mm
            showgrid=True,
            gridcolor="black",
            gridwidth=1.2,
            ticks="outside",
            ticklen=6,
            tickcolor="black",
            zeroline=False,
            tickfont=dict(size=10, color="white"),
        ),
        yaxis=dict(
            title=dict(text="Y [mm]", font=dict(size=12, color="white")),
            title_standoff=40,
            range=[-15, 275],
            tick0=0,
            dtick=50,                 # grid every 50 mm
            showgrid=True,
            gridcolor="black",
            gridwidth=1.2,
            ticks="outside",
            ticklen=6,
            tickcolor="black",
            zeroline=False,
            tickfont=dict(size=10, color="white"),
        ),
    )

    # Right-hand MRR axis (locked to plate y-axis)
    if y_ticks_axis2:
        fig.update_layout(
            yaxis2=dict(
                overlaying='y',
                side='right',
                range=[-15, 275],
                tickvals=y_ticks_axis2,
                ticktext=tick_labels_axis2,
                tickfont=dict(size=10, color="white"),
                showgrid=False,
                zeroline=False,
                matches='y',
                title=dict(
                    text="MRR Qw [mm³/min]",
                    font=dict(size=12, color="white"),
                ),
            )
        )

    # lock aspect ratio of plate; zooming keeps geometry and both y-axes aligned
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


# NOTE: currently unused — no callers found anywhere in the codebase. Kept pending review.
def plot_correlation_blocks_IWF(df_wide):
    """
    Simple IWF-style correlation heatmap with signals grouped into blocks
    using hierarchical clustering on |corr|.

    - computes Pearson correlation
    - reorders rows/cols so similar signals sit together
    - shows only upper triangle without diagonal with rounded values in each cell
    """

    # IWF template 
    register_templates()
    pio.templates.default = "IWF_template"

    # Correlation matrix 
    corr = df_wide.corr(method="pearson").fillna(0.0)
    corr = corr.round(1)              # for labels
    abs_corr = corr.abs()

    # Very simple clustering-based ordering 
    # distance = 1 - |corr|  → similar signals = small distance
    Z = linkage(1 - abs_corr, method="average")
    order = leaves_list(Z)

    corr_ord = corr.iloc[order, order]
    labels = corr_ord.columns.to_list()

    # Mask lower triangle 
    mask = np.triu(np.ones_like(corr_ord, dtype=bool), k=1)
    z = corr_ord.where(mask, np.nan)

    # Text labels (hide NaNs) 
    text_matrix = z.astype(str).replace("nan", "")

    # Heatmap 
    fig = go.Figure(
        data=go.Heatmap(
            z=z.values,
            x=labels,
            y=labels,
            colorscale=IWF_Red_Fade,
            zmin=-1, zmax=1,
            text=text_matrix.values,
            texttemplate="%{text}",
            textfont={"size": 8, "color": "black"},
            hovertemplate=(
                "Signal 1: %{x}<br>"
                "Signal 2: %{y}<br>"
                "r = %{z}<extra></extra>"
            ),
            colorbar=dict(title="r"),
        )
    )

    # Layout 
    fig.update_layout(
        title=dict(
            text="Correlation Blocks (Upper Triangle Only)<br>"
                 "<sub>Signals grouped by similarity of |r|</sub>",
            x=0.5, xanchor="center",
        ),
        width=900,
        height=900,
        template="IWF_template",
        xaxis=dict(tickangle=45, side="bottom", tickfont=dict(size=9)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
        margin=dict(l=80, r=80, t=120, b=120),
    )

    return fig, corr_ord

#########PCA###############  CAN I DELETE THIS?

# NOTE: currently unused — only called by plot_pca_biplot, which is itself unused.
# Effectively dead unless that function is reintroduced. Kept pending review.
def build_iwf_color_palette(n_colors=50):
    """
    Creates an extended color palette based solely on IWF's defined fades and brand colors.
    Returns up to n_colors visually distinct hues, all within the IWF identity.
    """
    fades = [
        IWF_Red_Fade,
        IWF_GreyBlue_fade,
        IWF_Brown_fade,
        IWF_Black_fade,
    ]
    
    # Sample ~8 tones from each fade gradient
    shades = []
    for fade in fades:
        shades += sample_colorscale(fade, [i/7 for i in range(8)])
    
    # Add core brand anchors (Fraunhofer + PTZ)
    base_colors = FraunhoferColors + PTZ_colors
    combined = shades + base_colors
    
    # Return the first n_colors unique tones
    return combined[:n_colors]
    
# NOTE: currently unused — no callers found anywhere in the codebase. Kept pending review.
def plot_pca_biplot(X_pca, loadings, explained_var, arrow_scale=5, nut_state=None):

    """
    PCA biplot showing:
      - samples colored by dominant signal
      - arrows for top 5 loadings per PC (PC1 & PC2)
    """
    register_templates()
    pio.templates.default = "IWF_template"
    
    if nut_state is not None:
            try:
                selected_path = nut_state.get("path", None)
                df = nut_state.get("df", None)
                if selected_path and df is not None:
                    print(f"✅ Using selected file: {selected_path.name} ({len(df):,} rows)")
            except Exception as e:
                print(f"⚠️ Could not read widget state: {e}")

    # Build IWF color palette 
    n_signals = len(loadings.index)
    color_list = build_iwf_color_palette(n_colors=n_signals)
    color_map = {sig: color_list[i % len(color_list)] for i, sig in enumerate(loadings.index)}

    # Normalize loading vectors 
    signal_vectors = loadings[["PC1", "PC2"]].values
    norms = np.linalg.norm(signal_vectors, axis=1)
    norms[norms == 0] = 1e-9
    signal_vectors = signal_vectors / norms[:, None]

    # Assign each sample to its dominant signal 
    signal_names = loadings.index.tolist()
    sample_labels = []
    for p in X_pca[:, :2]:
        scores = np.dot(signal_vectors, p)
        idx = np.argmax(np.abs(scores))
        sample_labels.append(signal_names[idx])

    # Scatter (samples) 
    fig = go.Figure()
    for sig in loadings.index:
        mask = np.array(sample_labels) == sig
        if not mask.any():
            continue
        fig.add_trace(go.Scattergl(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            mode="markers",
            name=sig,
            marker=dict(size=4, color=color_map[sig], opacity=0.8),
            hovertemplate=f"<b>{sig}</b><br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<extra></extra>"
        ))

    # Select top 5 loadings per PC 
    top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(5)
    top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(5)
    selected = top_pc1.index.union(top_pc2.index)
    top_loadings = loadings.loc[selected]

    # Draw arrows for selected loadings 
    for sig, row in top_loadings.iterrows():
        fig.add_annotation(
            x=row["PC1"] * arrow_scale,
            y=row["PC2"] * arrow_scale,
            ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=5,
            arrowcolor=color_map[sig],
            opacity=0.9,
            hovertext=f"{sig}<br>PC1: {row['PC1']:.3f}<br>PC2: {row['PC2']:.3f}"
        )
        fig.add_trace(go.Scatter(
            x=[row["PC1"] * arrow_scale],
            y=[row["PC2"] * arrow_scale],
            mode="text",
            text=[sig],
            textposition="top center",
            textfont=dict(size=10, color=color_map[sig]),
            showlegend=False
        ))

    # Layout 
    fig.update_layout(
        title="PCA Biplot — Samples + Top 5 Signals per PC",
        xaxis_title=f"PC1 ({explained_var[0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({explained_var[1]*100:.1f}% variance)",
        template="IWF_template",
        width=1050,
        height=800,
        plot_bgcolor="white",
        legend=dict(
            title=f"Signals ({n_signals} total)",
            orientation="v",
            x=1.05, y=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=0.5,
            font=dict(size=9)
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# NOTE: currently unused — no callers found anywhere in the codebase. Kept pending review.
def plot_top5_per_pc(loadings):
    """
    Plot vertical bars of top 5 absolute loadings for PC1 and PC2.
    """
    top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(5)
    top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(5)

    df_plot = pd.DataFrame({
        "Signal": top_pc1.index.tolist() + top_pc2.index.tolist(),
        "Loading": top_pc1.values.tolist() + top_pc2.values.tolist(),
        "PC": ["PC1"] * 5 + ["PC2"] * 5
    })

    fig = px.bar(
        df_plot,
        x="Signal",
        y="Loading",
        color="PC",
        barmode="group",
        title="Top 5 Signals per Principal Component",
        text_auto=".3f",
        template="IWF_template"
    )
    fig.update_layout(
        xaxis_title="Signal",
        yaxis_title="|Loading| magnitude",
        width=800,
        height=500
    )
    return fig