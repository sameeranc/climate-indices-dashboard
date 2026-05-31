"""
Sri Lanka Climate Indices Dashboard – Light Theme
====================================
Requirements:
    pip install dash dash-bootstrap-components plotly pandas numpy werkzeug flask

Run:
    python app_light.py
Then open http://127.0.0.1:8050

Admin login:
    username: admin       password: climate2024
    username: superuser   password: slclimate!

File naming convention (place files inside ./data/):
    12.5 km grid:
        temp_indices_monthly_12.5.csv
        temp_indices_annual_12.5.csv
        precip_indices_monthly_12.5.csv
        precip_indices_annual_12.5.csv
        drought_indices_12.5.csv
        grid_coords_12.5.csv
        grid_12.5.geojson          (or shapefile: 12.5_grid.shp / .dbf)

    25 km grid:
        temp_indices_monthly_25.csv
        temp_indices_annual_25.csv
        precip_indices_monthly_25.csv
        precip_indices_annual_25.csv
        drought_indices_25.csv
        grid_coords_25.csv
        grid_25.geojson            (or shapefile: 25_grid.shp / .dbf)
"""

import os, base64, json, zipfile, tempfile
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from werkzeug.security import generate_password_hash, check_password_hash

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS & AUTH
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

ADMIN_USERS = {
    "admin":     generate_password_hash("climate2024"),
    "superuser": generate_password_hash("slclimate!"),
}

# ─────────────────────────────────────────────────────────────────────────────
#  GRID RESOLUTION CONFIG
#  All file references go through these helpers so only one place to update.
# ─────────────────────────────────────────────────────────────────────────────
GRID_OPTIONS = [
    {"label": " 25 km Grid",   "value": "25"},
    {"label": " 12.5 km Grid", "value": "12.5"},
]

def grid_files(grid):
    """Return dict of all expected filenames for a given grid resolution."""
    g = grid  # "25" or "12.5"
    return {
        "temp_monthly":   f"temp_indices_monthly_{g}.csv",
        "temp_annual":    f"temp_indices_annual_{g}.csv",
        "precip_monthly": f"precip_indices_monthly_{g}.csv",
        "precip_annual":  f"precip_indices_annual_{g}.csv",
        "drought":        f"drought_indices_{g}.csv",
        "coords":         f"grid_coords_{g}.csv",
        "geojson":        f"grid_{g}.geojson",
        "shp":            f"{g}_grid.shp",
        "dbf":            f"{g}_grid.dbf",
    }

# ─────────────────────────────────────────────────────────────────────────────
#  METADATA
# ─────────────────────────────────────────────────────────────────────────────
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

TEMP_MONTHLY_INDICES = ["TX90p","TX10p","TN90p","TN10p","DTR","TXx","TXn","TNx","TNn"]
TEMP_ANNUAL_INDICES  = ["TX90p","TX10p","TN90p","TN10p","WSDI","DTR","TXx","TXn","TNx","TNn",
                        "rv20TXx","rv20TXn","rv20TNx","rv20TNn"]

INDEX_DESC = {
    "TX90p":  "Warm days - % days Tmax > 90th pct",
    "TX10p":  "Cool days - % days Tmax < 10th pct",
    "TN90p":  "Warm nights - % days Tmin > 90th pct",
    "TN10p":  "Cool nights - % days Tmin < 10th pct",
    "DTR":    "Diurnal Temperature Range",
    "TXx":    "Monthly max of daily max temp",
    "TXn":    "Monthly min of daily max temp",
    "TNx":    "Monthly max of daily min temp",
    "TNn":    "Monthly min of daily min temp",
    "WSDI":   "Warm Spell Duration Index",
    "rv20TXx":"20-yr return value of TXx",
    "rv20TXn":"20-yr return value of TXn",
    "rv20TNx":"20-yr return value of TNx",
    "rv20TNn":"20-yr return value of TNn",
}

INDEX_UNIT = {
    "TX90p":"%","TX10p":"%","TN90p":"%","TN10p":"%",
    "DTR":"degC","TXx":"degC","TXn":"degC","TNx":"degC","TNn":"degC",
    "WSDI":"days",
    "rv20TXx":"degC","rv20TXn":"degC","rv20TNx":"degC","rv20TNn":"degC",
}

def cscale(idx):
    warm = {"TX90p","TN90p","TXx","TNx","WSDI","rv20TXx","rv20TNx"}
    cool = {"TX10p","TN10p","TXn","TNn","rv20TXn","rv20TNn"}
    return "YlOrRd" if idx in warm else ("YlGnBu" if idx in cool else "Viridis")

# ─────────────────────────────────────────────────────────────────────────────
#  LIGHT COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#f0f4f8"
CARD    = "#ffffff"
BORDER  = "#d0dce8"
A1      = "#2563eb"
A2      = "#0891b2"
A3      = "#d97706"
TEXT    = "#1e293b"
SUB     = "#64748b"
OK      = "#059669"
ERR     = "#dc2626"
TAB_ON  = "#2563eb"
TAB_OFF = "#e2eaf2"

CARD_STYLE = {
    "backgroundColor": CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "14px",
    "padding": "20px",
    "boxShadow": "0 1px 6px rgba(30,41,59,0.07)",
}

# Accent colour per grid – blue for 25 km, violet for 12.5 km
GRID_COLOR = {"25": "#2563eb", "12.5": "#7c3aed"}

# ─────────────────────────────────────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_df(fname):
    p = os.path.join(DATA_DIR, fname)
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.replace(-99.0,  np.nan, inplace=True)
    df.replace(-99,    np.nan, inplace=True)
    df.replace(-999.0, np.nan, inplace=True)
    return df

def load_geojson(grid):
    p = os.path.join(DATA_DIR, grid_files(grid)["geojson"])
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def load_coords(grid):
    p = os.path.join(DATA_DIR, grid_files(grid)["coords"])
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["Grid","lat","lon"])

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def base_layout(h=400, **kw):
    d = dict(
        paper_bgcolor=CARD,
        plot_bgcolor="#f8fafc",
        font=dict(color=TEXT, family="'DM Sans','Segoe UI',sans-serif"),
        margin=dict(l=48, r=16, t=28, b=44),
        height=h,
        xaxis=dict(gridcolor="#e2eaf2", linecolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor="#e2eaf2", linecolor=BORDER, zerolinecolor=BORDER),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=BORDER, font=dict(color=TEXT)),
        legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(color=SUB, size=10)),
    )
    d.update(kw)
    return d

def empty_fig(h=400, msg="No data available"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=SUB, size=13))
    fig.update_layout(base_layout(h))
    return fig

def section_hdr(txt):
    return html.Div(txt, style={
        "fontSize": "0.71rem", "color": SUB,
        "textTransform": "uppercase", "letterSpacing": "0.08em",
        "marginBottom": "10px", "fontWeight": "700",
    })

def kpi_card(cid_val, cid_unit, title, icon, color):
    return html.Div([
        html.I(className=f"bi {icon}",
               style={"fontSize": "1.7rem", "color": color,
                      "marginBottom": "5px", "display": "block"}),
        html.Div(title, style={"fontSize": "0.69rem", "color": SUB,
                               "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        html.Span(id=cid_val,
                  style={"fontSize": "1.5rem", "fontWeight": "700", "color": TEXT}),
        html.Span(" ", style={"fontSize": "1rem"}),
        html.Span(id=cid_unit, style={"fontSize": "0.85rem", "color": SUB}),
    ], style={**CARD_STYLE, "flex": "1", "minWidth": "130px", "textAlign": "center"})

# ─────────────────────────────────────────────────────────────────────────────
#  UPLOAD PANEL  (one per grid resolution, swapped by auth+grid callbacks)
# ─────────────────────────────────────────────────────────────────────────────
US_BASE = {
    "width": "100%", "height": "62px", "lineHeight": "62px",
    "borderWidth": "1.5px", "borderStyle": "dashed", "borderRadius": "10px",
    "textAlign": "center", "color": SUB,
    "backgroundColor": "#f0f4f8", "cursor": "pointer",
    "marginBottom": "10px", "fontSize": "0.82rem",
}

def make_upload_panel(grid):
    gcolor = GRID_COLOR[grid]
    us = {**US_BASE, "borderColor": gcolor}
    ids = lambda k: f"up-{k}-{grid}"

    return html.Div([
        html.Div([
            html.I(className="bi bi-cloud-upload-fill me-2", style={"color": gcolor}),
            html.Span(f"Admin Data Panel  -  {grid} km Grid",
                      style={"fontWeight": "700", "color": gcolor, "fontSize": "1rem"}),
        ], style={"marginBottom": "16px"}),
        dbc.Row([
            dbc.Col([
                html.P("Temperature",
                       style={"color": SUB, "fontSize": "0.74rem",
                              "textTransform": "uppercase", "marginBottom": "6px"}),
                dcc.Upload(id=ids("temp-m"), children="Monthly CSV",  style=us, multiple=False),
                dcc.Upload(id=ids("temp-a"), children="Annual CSV",   style=us, multiple=False),
            ], md=4),
            dbc.Col([
                html.P("Precipitation",
                       style={"color": SUB, "fontSize": "0.74rem",
                              "textTransform": "uppercase", "marginBottom": "6px"}),
                dcc.Upload(id=ids("prec-m"), children="Monthly CSV",  style=us, multiple=False),
                dcc.Upload(id=ids("prec-a"), children="Annual CSV",   style=us, multiple=False),
            ], md=4),
            dbc.Col([
                html.P("Drought & Shapefile",
                       style={"color": SUB, "fontSize": "0.74rem",
                              "textTransform": "uppercase", "marginBottom": "6px"}),
                dcc.Upload(id=ids("drought"), children="Drought CSV",      style=us, multiple=False),
                dcc.Upload(id=ids("shape"),   children="Shapefile (.zip)", style=us, multiple=False),
            ], md=4),
        ]),
        html.Div(id=ids("status"),
                 style={"fontSize": "0.83rem", "marginTop": "4px", "minHeight": "20px"}),
    ], style={
        **CARD_STYLE,
        "marginBottom": "22px",
        "borderColor": gcolor,
        "borderWidth": "1.5px",
        "backgroundColor": "#f5f3ff" if grid == "12.5" else "#eff6ff",
    })

admin_panel_25  = make_upload_panel("25")
admin_panel_125 = make_upload_panel("12.5")

# ─────────────────────────────────────────────────────────────────────────────
#  LOGIN MODAL
# ─────────────────────────────────────────────────────────────────────────────
login_modal = dbc.Modal([
    dbc.ModalHeader(
        html.Span([
            html.I(className="bi bi-shield-lock-fill me-2", style={"color": A1}),
            "Admin Login",
        ], style={"color": TEXT, "fontWeight": "700"}),
        style={"backgroundColor": CARD, "borderColor": BORDER},
        close_button=True,
    ),
    dbc.ModalBody([
        dbc.Alert(id="login-alert", is_open=False, color="danger",
                  dismissable=True, style={"fontSize": "0.84rem"}),
        dbc.Label("Username", style={"color": SUB, "fontSize": "0.82rem"}),
        dbc.Input(id="login-user", type="text", placeholder="admin",
                  style={"backgroundColor": "#f8fafc", "color": TEXT,
                         "borderColor": BORDER, "marginBottom": "14px"}),
        dbc.Label("Password", style={"color": SUB, "fontSize": "0.82rem"}),
        dbc.Input(id="login-pass", type="password", placeholder="........",
                  style={"backgroundColor": "#f8fafc", "color": TEXT,
                         "borderColor": BORDER, "marginBottom": "22px"}),
        dbc.Button([html.I(className="bi bi-box-arrow-in-right me-2"), "Sign In"],
                   id="login-btn", color="primary", className="w-100",
                   style={"fontWeight": "700"}),
    ], style={"backgroundColor": CARD}),
], id="login-modal", is_open=False, centered=True)

# ─────────────────────────────────────────────────────────────────────────────
#  BUILD LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
def make_layout():
    # Bootstrap year range from the default grid (25 km)
    df_m = load_df(grid_files("25")["temp_monthly"])
    all_years = (sorted(df_m["Year"].unique().tolist())
                 if not df_m.empty else list(range(1981, 2025)))
    yr_min, yr_max = min(all_years), max(all_years)

    m_idx = ([c for c in df_m.columns if c not in ("Year","Month","Grid")]
             if not df_m.empty else TEMP_MONTHLY_INDICES)
    m_opts = [{"label": f"{k} - {INDEX_DESC.get(k,k)}", "value": k} for k in m_idx]

    # ── Navbar ────────────────────────────────────────────────────────────────
    navbar = dbc.Navbar(dbc.Container([
        html.Div([
            html.Img(
                src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/"
                    "Flag_of_Sri_Lanka.svg/36px-Flag_of_Sri_Lanka.svg.png",
                height="26px", style={"marginRight": "10px", "borderRadius": "2px"},
            ),
            html.Span("Sri Lanka Climate Dashboard",
                      style={"color": TEXT, "fontWeight": "800", "fontSize": "1.05rem"}),
            html.Span(" - Climate Indices Explorer 1981-2024",
                      style={"color": SUB, "fontSize": "0.79rem", "marginLeft": "8px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Div(id="auth-badge"),
            dbc.Button(
                [html.I(className="bi bi-person-lock me-1"), "Admin Login"],
                id="open-login", size="sm", outline=True, color="primary",
                style={"fontSize": "0.78rem", "padding": "4px 14px"}, n_clicks=0,
            ),
            dbc.Button(
                [html.I(className="bi bi-box-arrow-right me-1"), "Log Out"],
                id="logout-btn", size="sm", outline=True, color="danger",
                style={"fontSize": "0.78rem", "padding": "4px 14px", "display": "none"},
                n_clicks=0,
            ),
        ], style={"display": "flex", "alignItems": "center",
                  "gap": "12px", "marginLeft": "auto"}),
    ], fluid=True),
    color=CARD, dark=False,
    style={"borderBottom": f"1px solid {BORDER}", "padding": "6px 0",
           "boxShadow": "0 1px 8px rgba(30,41,59,0.10)"})

    # ── Grid Resolution selector ──────────────────────────────────────────────
    grid_selector = html.Div([
        html.I(className="bi bi-grid-3x3 me-2",
               style={"color": A1, "fontSize": "1.1rem"}),
        html.Span("Grid Resolution",
                  style={"fontWeight": "700", "color": TEXT,
                         "fontSize": "0.92rem", "marginRight": "24px"}),
        dcc.RadioItems(
            id="grid-res",
            value="25",
            options=GRID_OPTIONS,
            inline=True,
            labelStyle={
                "marginRight": "28px", "fontSize": "0.93rem",
                "fontWeight": "600", "cursor": "pointer", "color": TEXT,
            },
            inputStyle={"marginRight": "6px", "accentColor": A1,
                        "width": "15px", "height": "15px"},
        ),
        html.Span(id="grid-badge", children="25 km", style={
            "marginLeft": "8px", "fontSize": "0.78rem", "fontWeight": "700",
            "padding": "3px 12px", "borderRadius": "20px",
            "backgroundColor": "#dbeafe", "color": A1,
        }),
    ], style={
        **CARD_STYLE,
        "marginBottom": "20px",
        "padding": "14px 22px",
        "display": "flex",
        "alignItems": "center",
        "borderLeft": f"4px solid {A1}",
    })

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_style = {
        "backgroundColor": TAB_OFF, "color": SUB, "border": "none",
        "padding": "10px 24px", "fontSize": "0.9rem",
        "borderRadius": "8px 8px 0 0", "margin": "0 2px",
    }
    tab_sel = {
        "backgroundColor": TAB_ON, "color": "#ffffff", "border": "none",
        "padding": "10px 24px", "fontSize": "0.9rem", "fontWeight": "700",
        "borderRadius": "8px 8px 0 0", "margin": "0 2px",
    }

    tabs = dcc.Tabs(id="main-tabs", value="temp", children=[
        dcc.Tab(label="  Temperature",   value="temp",   style=tab_style, selected_style=tab_sel),
        dcc.Tab(label="  Precipitation", value="precip", style=tab_style, selected_style=tab_sel),
        dcc.Tab(label="  Drought",       value="drought",style=tab_style, selected_style=tab_sel),
    ], style={"marginBottom": "22px", "borderBottom": f"2px solid {BORDER}"})

    # ── Temperature controls ──────────────────────────────────────────────────
    dd_style = {"backgroundColor": "#f8fafc", "color": TEXT, "border": f"1px solid {BORDER}"}

    temp_controls = html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Resolution",
                           style={"color": SUB, "fontSize": "0.76rem", "marginBottom": "3px"}),
                dcc.RadioItems(id="t-res", value="monthly",
                    options=[{"label": " Monthly", "value": "monthly"},
                             {"label": " Annual",  "value": "annual"}],
                    inline=True,
                    labelStyle={"marginRight": "16px", "color": TEXT, "fontSize": "0.88rem"},
                    inputStyle={"marginRight": "4px", "accentColor": A1}),
            ], md=2),
            dbc.Col([
                html.Label("Index",
                           style={"color": SUB, "fontSize": "0.76rem", "marginBottom": "3px"}),
                dcc.Dropdown(id="t-idx", options=m_opts,
                    value=m_opts[0]["value"] if m_opts else None,
                    clearable=False, style=dd_style),
            ], md=4),
            dbc.Col([
                html.Label("Month",
                           style={"color": SUB, "fontSize": "0.76rem", "marginBottom": "3px"}),
                dcc.Dropdown(id="t-month",
                    options=[{"label": v, "value": k} for k, v in MONTH_NAMES.items()],
                    value=1, clearable=False, style=dd_style),
            ], md=2, id="t-month-col"),
            dbc.Col([
                html.Label("Year Range",
                           style={"color": SUB, "fontSize": "0.76rem", "marginBottom": "3px"}),
                dcc.RangeSlider(id="t-years",
                    min=yr_min, max=yr_max, value=[yr_min, yr_max],
                    marks={y: {"label": str(y),
                               "style": {"fontSize": "0.68rem", "color": SUB, "marginTop": "4px"}}
                           for y in all_years if y % 10 == 0},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode="mouseup"),
            ], md=4),
        ], className="g-3 align-items-end"),
    ], style={"marginBottom": "20px"})

    # ── KPI row ───────────────────────────────────────────────────────────────
    kpi_row = html.Div([
        kpi_card("kpi-mean","kpi-mean-u","Mean",    "bi-activity",               A1),
        kpi_card("kpi-max", "kpi-max-u", "Maximum", "bi-arrow-up-circle-fill",   ERR),
        kpi_card("kpi-min", "kpi-min-u", "Minimum", "bi-arrow-down-circle-fill", A2),
        kpi_card("kpi-std", "kpi-std-u", "Std Dev", "bi-distribute-vertical",    A3),
        kpi_card("kpi-cnt", "kpi-cnt-u", "Grids",   "bi-grid-3x3",              SUB),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "20px"})

    # ── Charts ────────────────────────────────────────────────────────────────
    temp_charts = html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                section_hdr("Spatial Distribution (grid choropleth)"),
                dcc.Graph(id="t-map", figure=empty_fig(480, "Loading..."),
                          config={"scrollZoom": True, "displayModeBar": True,
                                  "modeBarButtonsToRemove": ["lasso2d","select2d"]}),
            ], style=CARD_STYLE), md=7),
            dbc.Col(html.Div([
                section_hdr("Annual Trend (area average)"),
                dcc.Graph(id="t-trend", figure=empty_fig(480, "Loading..."),
                          config={"displayModeBar": False}),
            ], style=CARD_STYLE), md=5),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(html.Div([
                section_hdr("Monthly Climatology (grid average)"),
                dcc.Graph(id="t-clim", figure=empty_fig(300, "Loading..."),
                          config={"displayModeBar": False}),
            ], style=CARD_STYLE), md=6),
            dbc.Col(html.Div([
                section_hdr("Annual Anomaly vs 1981-2010 baseline"),
                dcc.Graph(id="t-anom", figure=empty_fig(300, "Loading..."),
                          config={"displayModeBar": False}),
            ], style=CARD_STYLE), md=6),
        ], className="g-3"),
    ])

    # ── Placeholder panels ────────────────────────────────────────────────────
    def placeholder(name, icon, indices):
        return html.Div([
            html.I(className=f"bi {icon}", style={"fontSize": "3.5rem", "color": BORDER}),
            html.H4(f"{name} Indices",
                    style={"color": TEXT, "marginTop": "18px", "marginBottom": "10px"}),
            html.P("No data files uploaded yet.", style={"color": SUB}),
            html.P(f"Expected indices: {', '.join(indices)}",
                   style={"color": SUB, "fontSize": "0.83rem",
                          "maxWidth": "600px", "margin": "8px auto 0"}),
            html.Hr(style={"borderColor": BORDER, "margin": "22px auto", "width": "40%"}),
            html.Span([html.I(className="bi bi-shield-lock me-1"),
                       " Admins can upload data via the Admin Panel (top-right login)"],
                      style={"color": SUB, "fontSize": "0.83rem"}),
        ], style={
            "textAlign": "center", "padding": "80px 20px",
            **CARD_STYLE, "minHeight": "480px",
            "display": "flex", "flexDirection": "column",
            "alignItems": "center", "justifyContent": "center",
        })

    precip_ph = placeholder("Precipitation","bi-cloud-rain-heavy",
        ["PRCPTOT","Rx1day","Rx5day","SDII","R10mm","R20mm","CWD","CDD","R95p","R99p"])
    drought_ph = placeholder("Drought","bi-exclamation-triangle",
        ["SPI-3","SPI-6","SPI-12","SPEI-3","SPEI-6","SPEI-12","PDSI","scPDSI"])

    # ── Full layout ───────────────────────────────────────────────────────────
    return html.Div([
        dcc.Store(id="auth-store",  data={"ok": False, "user": ""}),
        dcc.Store(id="grid-store",  data="25"),
        dcc.Store(id="data-reload", data=0),

        login_modal,
        navbar,

        dbc.Container([
            html.Div(id="admin-area"),

            grid_selector,   # <-- Grid Resolution radio sits here, above tabs
            tabs,

            html.Div(id="tab-temp-content", children=[
                temp_controls, kpi_row, temp_charts,
            ]),
            html.Div(id="tab-precip-content", children=precip_ph,
                     style={"display": "none"}),
            html.Div(id="tab-drought-content", children=drought_ph,
                     style={"display": "none"}),

        ], fluid=True, style={"paddingTop": "18px", "paddingBottom": "50px"}),

        html.Footer(
            dbc.Container(
                html.Span(
                    "Sri Lanka Department of Meteorology  -  Climate Indices Dashboard  -  Data: 1981-2024",
                    style={"color": SUB, "fontSize": "0.74rem"},
                ),
                fluid=True,
            ),
            style={"borderTop": f"1px solid {BORDER}", "padding": "14px 0",
                   "textAlign": "center", "marginTop": "40px", "backgroundColor": CARD},
        ),
    ], style={
        "backgroundColor": BG, "minHeight": "100vh",
        "fontFamily": "'DM Sans','Segoe UI',system-ui,sans-serif", "color": TEXT,
    })

# ─────────────────────────────────────────────────────────────────────────────
#  APP INIT
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=False,
    title="Sri Lanka Climate Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width,initial-scale=1"}])
server = app.server
app.layout = make_layout

# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACK -- TAB VISIBILITY
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-temp-content",   "style"),
    Output("tab-precip-content", "style"),
    Output("tab-drought-content","style"),
    Input("main-tabs","value"),
)
def switch_tab(tab):
    show, hide = {"display":"block"}, {"display":"none"}
    return (
        show if tab=="temp"    else hide,
        show if tab=="precip"  else hide,
        show if tab=="drought" else hide,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACK -- GRID RESOLUTION selector
#  Updates: store, badge text/colour, and year-range slider bounds
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("grid-store", "data"),
    Output("grid-badge", "children"),
    Output("grid-badge", "style"),
    Output("t-years",    "min"),
    Output("t-years",    "max"),
    Output("t-years",    "value"),
    Output("t-years",    "marks"),
    Input("grid-res", "value"),
)
def update_grid(grid):
    df = load_df(grid_files(grid)["temp_monthly"])
    if not df.empty and "Year" in df.columns:
        years = sorted(df["Year"].dropna().astype(int).unique().tolist())
    else:
        years = list(range(1981, 2025))
    yr_min, yr_max = min(years), max(years)
    marks = {y: {"label": str(y),
                 "style": {"fontSize": "0.68rem", "color": SUB, "marginTop": "4px"}}
             for y in years if y % 10 == 0}

    gcolor = GRID_COLOR[grid]
    badge_style = {
        "marginLeft": "8px", "fontSize": "0.78rem", "fontWeight": "700",
        "padding": "3px 12px", "borderRadius": "20px",
        "backgroundColor": "#ede9fe" if grid == "12.5" else "#dbeafe",
        "color": gcolor,
    }
    return grid, f"{grid} km", badge_style, yr_min, yr_max, [yr_min, yr_max], marks

# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS -- AUTH
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("login-modal","is_open"),
    Input("open-login","n_clicks"),
    State("login-modal","is_open"),
    prevent_initial_call=True,
)
def open_modal(n, _): return True

@app.callback(
    Output("auth-store","data"),
    Output("login-alert","children"),
    Output("login-alert","is_open"),
    Output("login-modal","is_open", allow_duplicate=True),
    Input("login-btn","n_clicks"),
    State("login-user","value"),
    State("login-pass","value"),
    prevent_initial_call=True,
)
def do_login(n, user, pw):
    if not user or not pw:
        return no_update, "Please fill in both fields.", True, True
    user = user.strip()
    if user in ADMIN_USERS and check_password_hash(ADMIN_USERS[user], pw):
        return {"ok": True, "user": user}, "", False, False
    return {"ok": False, "user": ""}, "Invalid username or password.", True, True

@app.callback(
    Output("auth-store","data", allow_duplicate=True),
    Input("logout-btn","n_clicks"),
    prevent_initial_call=True,
)
def do_logout(n):
    return {"ok": False, "user": ""}

@app.callback(
    Output("auth-badge", "children"),
    Output("open-login", "style"),
    Output("logout-btn", "style"),
    Output("admin-area", "children"),
    Input("auth-store",  "data"),
    Input("grid-store",  "data"),
)
def update_auth(auth, grid):
    btn_show = {"fontSize": "0.78rem", "padding": "4px 14px"}
    btn_hide = {"fontSize": "0.78rem", "padding": "4px 14px", "display": "none"}
    if auth and auth.get("ok"):
        badge = html.Span([
            html.I(className="bi bi-person-check-fill me-1", style={"color": OK}),
            html.Span(auth["user"],
                      style={"color": OK, "fontWeight": "600", "fontSize": "0.84rem"}),
        ])
        # Show the upload panel that matches the active grid
        panel = admin_panel_125 if grid == "12.5" else admin_panel_25
        return badge, btn_hide, btn_show, panel
    return None, btn_show, btn_hide, None

# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS -- FILE UPLOADS  (one per grid, generated via factory function)
# ─────────────────────────────────────────────────────────────────────────────
def _save_upload(contents, filename, grid):
    if not contents or not filename:
        return None
    _, data = contents.split(",", 1)
    raw = base64.b64decode(data)

    if filename.lower().endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp:
            zp = os.path.join(tmp, "up.zip")
            with open(zp, "wb") as f: f.write(raw)
            with zipfile.ZipFile(zp) as zf: zf.extractall(DATA_DIR)
        _rebuild_geojson(grid)
        return f"Shapefile extracted ({grid} km)"

    # Auto-append grid suffix if not already present
    name, ext = os.path.splitext(filename)
    suffix = f"_{grid}"
    if not name.endswith(suffix):
        filename = f"{name}{suffix}{ext}"

    dest = os.path.join(DATA_DIR, filename)
    with open(dest, "wb") as f: f.write(raw)
    return f"Saved: {filename}"

def _rebuild_geojson(grid):
    shp = os.path.join(DATA_DIR, grid_files(grid)["shp"])
    dbf = os.path.join(DATA_DIR, grid_files(grid)["dbf"])
    if not (os.path.exists(shp) and os.path.exists(dbf)):
        return
    # Full SHP->GeoJSON conversion would go here if needed

def _register_upload_callback(grid):
    """Register the upload callback for a specific grid resolution."""
    ids = lambda k: f"up-{k}-{grid}"

    @app.callback(
        Output(ids("status"), "children"),
        Output("data-reload", "data", allow_duplicate=True),
        Input(ids("temp-m"),  "contents"), Input(ids("temp-a"),  "contents"),
        Input(ids("prec-m"),  "contents"), Input(ids("prec-a"),  "contents"),
        Input(ids("drought"), "contents"), Input(ids("shape"),   "contents"),
        State(ids("temp-m"),  "filename"), State(ids("temp-a"),  "filename"),
        State(ids("prec-m"),  "filename"), State(ids("prec-a"),  "filename"),
        State(ids("drought"), "filename"), State(ids("shape"),   "filename"),
        State("data-reload",  "data"),
        prevent_initial_call=True,
    )
    def handle_uploads(c1,c2,c3,c4,c5,c6,f1,f2,f3,f4,f5,f6,rev):
        pairs = [(c1,f1),(c2,f2),(c3,f3),(c4,f4),(c5,f5),(c6,f6)]
        msgs  = [m for m in
                 [_save_upload(c, f, grid) for c, f in pairs if c and f]
                 if m]
        status = html.Span("  |  ".join(msgs), style={"color": OK}) if msgs else ""
        return status, (rev or 0) + 1

_register_upload_callback("25")
_register_upload_callback("12.5")

# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACK -- INDEX DROPDOWN  (monthly/annual switch, grid-aware)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("t-idx",       "options"),
    Output("t-idx",       "value"),
    Output("t-month-col", "style"),
    Input("t-res",     "value"),
    Input("grid-store","data"),
    Input("data-reload","data"),
)
def switch_resolution(res, grid, _reload):
    key   = "temp_monthly" if res == "monthly" else "temp_annual"
    fname = grid_files(grid)[key]
    df    = load_df(fname)
    if df.empty:
        idxs = TEMP_MONTHLY_INDICES if res == "monthly" else TEMP_ANNUAL_INDICES
    else:
        idxs = [c for c in df.columns if c not in ("Year","Month","Grid")]
    opts = [{"label": f"{k} - {INDEX_DESC.get(k,k)}", "value": k} for k in idxs]
    val  = opts[0]["value"] if opts else None
    return opts, val, ({} if res == "monthly" else {"display": "none"})

# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACK -- TEMPERATURE CHARTS  (grid-aware main workhorse)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-mean","children"), Output("kpi-mean-u","children"),
    Output("kpi-max", "children"), Output("kpi-max-u", "children"),
    Output("kpi-min", "children"), Output("kpi-min-u", "children"),
    Output("kpi-std", "children"), Output("kpi-std-u", "children"),
    Output("kpi-cnt", "children"), Output("kpi-cnt-u", "children"),
    Output("t-map",   "figure"),
    Output("t-trend", "figure"),
    Output("t-clim",  "figure"),
    Output("t-anom",  "figure"),
    Input("t-idx",    "value"),
    Input("t-res",    "value"),
    Input("t-month",  "value"),
    Input("t-years",  "value"),
    Input("grid-store","data"),
    Input("data-reload","data"),
)
def update_temp(idx, res, month, yr_range, grid, _reload):
    unit   = INDEX_UNIT.get(idx, "")
    na     = "-"
    gcolor = GRID_COLOR.get(grid, A1)

    if not idx:
        return (na,"") * 5 + (empty_fig(480),) * 4

    key   = "temp_monthly" if res == "monthly" else "temp_annual"
    fname = grid_files(grid)[key]
    df    = load_df(fname)

    if df.empty or idx not in df.columns:
        return (na,"") * 5 + (
            empty_fig(480, f"No data for {grid} km grid - upload via Admin Panel"),
        ) * 4

    dff = df[(df["Year"] >= yr_range[0]) & (df["Year"] <= yr_range[1])].copy()
    dff_sel = (dff[dff["Month"] == (month or 1)]
               if res == "monthly" and "Month" in dff.columns else dff)

    vals = dff_sel[idx].replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return (na,"") * 5 + (empty_fig(480, "No valid data for selection"),) * 4

    fmt = lambda v: f"{v:.2f}"
    kpi_out = (
        fmt(vals.mean()), unit,
        fmt(vals.max()),  unit,
        fmt(vals.min()),  unit,
        fmt(vals.std()),  unit,
        str(dff_sel["Grid"].nunique()), "grids",
    )

    cs     = cscale(idx)
    map_agg = dff_sel.groupby("Grid")[idx].mean().reset_index()
    gj      = load_geojson(grid)
    coords  = load_coords(grid)

    # ── MAP ───────────────────────────────────────────────────────────────────
    if gj is not None:
        fig_map = go.Figure(go.Choroplethmapbox(
            geojson=gj,
            locations=map_agg["Grid"].astype(str),
            z=map_agg[idx],
            featureidkey="id",
            colorscale=cs, showscale=True,
            marker_opacity=0.85, marker_line_width=0.6,
            marker_line_color="rgba(255,255,255,0.5)",
            colorbar=dict(
                title=dict(text=unit, font=dict(color=SUB, size=11)),
                tickfont=dict(color=SUB, size=10),
                thickness=14, len=0.75,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=BORDER, borderwidth=1,
            ),
            hovertemplate=f"<b>Grid %{{location}}</b><br>{idx}: %{{z:.2f}} {unit}<extra></extra>",
        ))
        fig_map.update_layout(
            mapbox=dict(style="carto-positron",
                        center=dict(lat=7.87, lon=80.77), zoom=5.8),
            paper_bgcolor=CARD, height=480,
            font=dict(color=TEXT), margin=dict(l=0,r=0,t=0,b=0))
    elif not coords.empty:
        mp = map_agg.merge(coords, on="Grid", how="left").dropna(subset=["lat","lon"])
        # Smaller dots for the denser 12.5 km grid
        dot_size = 10 if grid == "12.5" else 14
        fig_map = go.Figure(go.Scattermapbox(
            lat=mp["lat"], lon=mp["lon"], mode="markers",
            marker=dict(size=dot_size, color=mp[idx], colorscale=cs, showscale=True,
                        colorbar=dict(title=unit, tickfont=dict(color=SUB))),
            text=mp["Grid"].astype(str),
            hovertemplate=f"Grid %{{text}}<br>{idx}: %{{marker.color:.2f}} {unit}<extra></extra>",
        ))
        fig_map.update_layout(
            mapbox=dict(style="carto-positron",
                        center=dict(lat=7.87, lon=80.77), zoom=5.5),
            paper_bgcolor=CARD, height=480,
            font=dict(color=TEXT), margin=dict(l=0,r=0,t=0,b=0))
    else:
        fig_map = empty_fig(480,
            f"Shapefile not found for {grid} km grid - upload via Admin Panel")

    # ── TREND ─────────────────────────────────────────────────────────────────
    if res == "monthly" and "Month" in dff.columns:
        td = dff[dff["Month"] == (month or 1)].groupby("Year")[idx].mean().reset_index()
    else:
        td = dff.groupby("Year")[idx].mean().reset_index()
    td = td.dropna(subset=[idx])

    z     = np.polyfit(td["Year"], td[idx], 1) if len(td) > 2 else [0, td[idx].mean()]
    ty    = np.poly1d(z)(td["Year"])
    slope = f"{'up' if z[0]>0 else 'down'} {abs(z[0]):.4f} {unit}/yr"

    fill_rgba = "rgba(124,58,237,0.08)" if grid == "12.5" else "rgba(37,99,235,0.08)"
    fig_trend = go.Figure([
        go.Scatter(x=td["Year"], y=td[idx], mode="lines+markers", name=idx,
                   line=dict(color=gcolor, width=2.2), marker=dict(size=4, color=gcolor),
                   fill="tozeroy", fillcolor=fill_rgba,
                   hovertemplate="%{x}: %{y:.2f} " + unit + "<extra></extra>"),
        go.Scatter(x=td["Year"], y=ty, mode="lines", name=f"Trend ({slope})",
                   line=dict(color=A3, width=1.8, dash="dot"), hoverinfo="skip"),
    ])
    fig_trend.update_layout(base_layout(480,
        xaxis=dict(title="Year", gridcolor="#e2eaf2"),
        yaxis=dict(title=f"{idx} ({unit})", gridcolor="#e2eaf2"),
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right",
                    font=dict(size=10, color=SUB)),
        hovermode="x unified"))

    # ── CLIMATOLOGY ───────────────────────────────────────────────────────────
    if res == "monthly" and "Month" in dff.columns:
        cd = dff.groupby("Month")[idx].mean().reset_index()
        cd["lbl"] = cd["Month"].map(MONTH_NAMES)
        xv, xl = cd["lbl"], "Month"
    else:
        cd = dff.groupby("Year")[idx].mean().reset_index()
        xv, xl = cd["Year"], "Year"

    fig_clim = go.Figure(go.Bar(
        x=xv, y=cd[idx],
        marker=dict(color=cd[idx], colorscale=cs, showscale=False, line=dict(width=0)),
        hovertemplate="%{x}: %{y:.2f} " + unit + "<extra></extra>"))
    fig_clim.update_layout(base_layout(300,
        bargap=0.15,
        xaxis=dict(title=xl, gridcolor="#e2eaf2"),
        yaxis=dict(title=f"{idx} ({unit})", gridcolor="#e2eaf2")))

    # ── ANOMALY ───────────────────────────────────────────────────────────────
    ad = dff.groupby("Year")[idx].mean().reset_index().dropna(subset=[idx])
    base_v = ad[(ad["Year"] >= 1981) & (ad["Year"] <= 2010)][idx].mean()
    ad["anom"] = ad[idx] - base_v

    fig_anom = go.Figure(go.Bar(
        x=ad["Year"], y=ad["anom"],
        marker=dict(color=[ERR if v >= 0 else A2 for v in ad["anom"]], line=dict(width=0)),
        hovertemplate="%{x}: %{y:+.2f} " + unit + "<extra></extra>"))
    fig_anom.add_hline(y=0, line_color=SUB, line_width=1, line_dash="dot")
    fig_anom.update_layout(base_layout(300,
        xaxis=dict(gridcolor="#e2eaf2"),
        yaxis=dict(title=f"Anomaly ({unit})", gridcolor="#e2eaf2", zeroline=False)))

    return kpi_out + (fig_map, fig_trend, fig_clim, fig_anom)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)