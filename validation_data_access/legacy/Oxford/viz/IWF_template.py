# viz/IWF_template.py

# Importieren der benötigten Plotly-Module
import plotly.graph_objects as go
import plotly.io as pio


# Basis-Template als Objekt (NICHT gleich registrieren!)
IWF_template_raw = go.layout.Template()

# Definition allgemeiner Konstanten für die Gestaltung
# Schriftart-Definition
Font = dict(family="Arial", size=14, color="black")

# Definition von Farbschemata
# Kategoriale Farben ohne Weiß
iwfColors_without_white = [
    "black",
    "rgb(159, 182, 196)",
    "rgb(125, 102, 102)",
    "rgb(153, 0, 0)",
]
# Farben des Fraunhofer-Instituts
FraunhoferColors = [
    "rgb(0, 152, 121)",
    "rgb(0, 153, 178)",
    "rgb(67, 105, 123)",
    "rgb(97, 101, 103)",
    "rgb(147, 151, 153)",
    "rgb(199, 201, 202)",
]
# Kombinierte Farbpalette
PTZ_colors = [
    "#000000",  # schwarz
    "#9fb6c4",  # blau original
    "#990000",  # rot original
    "#009879",  # Fraunhofer grün original
    "#43697b",  # Fraunhofer dunkelblau original
    "#7d6666",  # braun original
    "#616567",  # grau original
    "#73bba4",  # grün hell
    "#bb5555",  # rot medium
    "#79bacb",  # Fraunhofer light bue
    "#bfced7",  # blau medium
    "#a79696",  # braun medium
    "#baddd0",  # Fraunhofer light green
    "#ddaaaa",  # light red
    "#bedce5",  #  Fraunhofer blue medium

]

# Definition von sequenziellen Farbverläufen
IWF_Red_Fade = ["#ffe6e6", "#990000"]
IWF_GreyBlue_fade = ["#dfe7ec", "#9fb6c4", "#3a515f"]
IWF_Brown_fade = ["#e8e3e3", "#7d6666", "#382e2e"]
IWF_Black_fade = ["#f2f2f2", "#000000"]

# Definition von Markersymbolen
# Symbole für 2D-Diagramme
marker_symbol = [
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "triangle-down",
    "cross",
    "x",
]
# Symbole für 3D-Diagramme
marker_symbol_3d = [
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "circle-open",
    "diamond-open",
    "square-open",
]

# Vorbereitung der Daten für verschiedene Diagrammtypen
# Initialisierung von Listen für Scatter-, Scattergl- und Scatter3d-Diagramme
scatter, scatter_gl, scatter_3d = [], [], []
for i, c in enumerate(PTZ_colors):
    # Hinzufügen von Scatter-Elementen mit individuellen Markern und Hoverlabels
    scatter.append(
        go.Scatter(
            marker=dict(symbol=marker_symbol[i % len(marker_symbol)], size=8),
            hoverlabel=dict(
                bgcolor="white", bordercolor=PTZ_colors[i % len(PTZ_colors)], font=Font
            ),
        )
    )

    scatter_gl.append(
        go.Scattergl(
            marker=dict(symbol=marker_symbol[i % len(marker_symbol)], size=8),
            hoverlabel=dict(
                bgcolor="white", bordercolor=PTZ_colors[i % len(PTZ_colors)], font=Font
            ),
        )
    )

    scatter_3d.append(
        go.Scatter3d(
            marker=dict(symbol=marker_symbol_3d[i % len(marker_symbol_3d)], size=8),
            hoverlabel=dict(
                bgcolor="white", bordercolor=PTZ_colors[i % len(PTZ_colors)], font=Font
            ),
        )
    )

# Zuweisung der vorbereiteten Daten zum Template
IWF_template_raw.data = dict(
    scatter=scatter, scattergl=scatter_gl, scatter3d=scatter_3d
)

# Konfiguration des Layouts
# Achsenlayout für 2D-Diagramme
axis_layout = dict(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    gridcolor="black",
    gridwidth=1,
    zeroline=True,
    zerolinewidth=1.5,
    zerolinecolor="black",
    ticks="outside",
    tickcolor="rgba(0 ,0, 0, 0)",
    ticklen=0,
    tickwidth=0,
    title=dict(font=Font),
)

# Achsenlayout für 3D-Diagramme
axis_layout_3d = dict(
    backgroundcolor="white",
    gridcolor="black",
    gridwidth=1,
    linecolor="black",
    showbackground=True,
    ticks="outside",
    zerolinecolor="black",
    zerolinewidth=1.52,
    tickcolor="rgba(0 ,0, 0, 0)",
    ticklen=0,
    tickwidth=0,
    title=dict(font=Font),
)

# Definition von Formen für das Layout
shapes = [
    dict(
        name="black_frame",
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1.0,
        y1=1.0,
        line=dict(color="black", width=1),
    )
]

# Festlegen des Layouts im Template
IWF_template_raw.layout = dict(
    font=Font,
    title=dict(font=Font),
    legend=dict(font=Font),
    plot_bgcolor="white",
    xaxis=axis_layout,
    yaxis=axis_layout,
    margin=dict(l=10, r=10, b=10, t=30, pad=0),  # Festlegen der Ränder
    width=600,
    height=400,  # Festlegen der Diagrammgröße
    scene=dict(xaxis=axis_layout_3d, yaxis=axis_layout_3d, zaxis=axis_layout_3d),
    shapes=shapes,
    colorway=PTZ_colors,
    colorscale=dict(sequential=IWF_Red_Fade),
)

def register_templates():
    """
    Registriert IWF-spezifische Plotly-Templates im globalen Template-Registry.

    Templates:
    - IWF_template_raw: individuelles Layout (als Objekt)
    - IWF_template: Kombination aus 'plotly' + IWF_template_raw (als String)
    """
    if "IWF_template_raw" not in pio.templates:
        pio.templates["IWF_template_raw"] = IWF_template_raw
    if "IWF_template" not in pio.templates:
        # Hinweis: hier wird NICHT das Template-Objekt kombiniert, sondern Template-Namen!
        pio.templates["IWF_template"] = "plotly+IWF_template_raw"


# pio.templates.default = "IWF_template"
