import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set up page configurations
st.set_page_config(page_title="Climate Indices Dashboard", layout="wide", page_icon="🌍")

st.title("🌍 Climate Indices Visualization Dashboard")
st.markdown("""
This dashboard visualizes annual temperature-related climate indices from **1981 to 2024** across different grid cells. 
Select a specific grid cell or inspect the global average using the sidebar menu.
""")

# Load and clean data
@st.cache_data
def load_data():
    # Adjust path if needed; assuming file is in data/ or same directory
    try:
        df = pd.read_csv("data/temp_indices_annual.csv")
    except FileNotFoundError:
        df = pd.read_csv("temp_indices_annual.csv")
        
    # Data Cleaning: Replace the -99 placeholder with NaN for temperature indices
    cols_to_clean = ['TXx', 'TXn', 'TNx', 'TNn']
    for col in cols_to_clean:
        df[col] = df[col].replace(-99, np.nan)
    return df

df = load_data()

# Dictionary containing definitions for user clarity
index_descriptions = {
    "TX90p": "Warm days (% of days where Maximum Temperature > 90th percentile)",
    "TX10p": "Cold days (% of days where Maximum Temperature < 10th percentile)",
    "TN90p": "Warm nights (% of days where Minimum Temperature > 90th percentile)",
    "TN10p": "Cold nights (% of days where Minimum Temperature < 10th percentile)",
    "WSDI": "Warm Spell Duration Index (Count of days in intervals of at least 6 consecutive warm days)",
    "DTR": "Diurnal Temperature Range (Mean of monthly maximum minus minimum temperatures)",
    "TXx": "Max value of daily maximum temperature (°C)",
    "TXn": "Min value of daily maximum temperature (°C)",
    "TNx": "Max value of daily minimum temperature (°C)",
    "TNn": "Min value of daily minimum temperature (°C)"
}

# Sidebar Setup
st.sidebar.header("🎛️ Control Panel")

# Grid filter
grid_options = ["All Grids (Average)"] + sorted(df['Grid'].unique().tolist())
selected_grid = st.sidebar.selectbox("Select Grid Location:", grid_options)

# Primary Index filter
selected_index = st.sidebar.selectbox("Select Primary Index:", list(index_descriptions.keys()))
st.sidebar.info(f"**Description:** {index_descriptions[selected_index]}")

# Process DataFrame based on selection
if selected_grid == "All Grids (Average)":
    filtered_df = df.groupby('Year')[selected_index].mean().reset_index()
    plot_title = f"Global Grid Average of {selected_index} Over Time"
else:
    filtered_df = df[df['Grid'] == selected_grid]
    plot_title = f"{selected_index} Trend for Grid Cell {selected_grid}"

# --- MAIN DASHBOARD LAYOUT ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(plot_title)
    # Interactive Time-Series Line Plot
    fig = px.line(
        filtered_df, 
        x="Year", 
        y=selected_index, 
        markers=True,
        labels={"Year": "Year", selected_index: selected_index},
        template="plotly_white"
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Summary Statistics")
    stats = filtered_df[selected_index].describe()
    st.dataframe(stats.to_frame().style.format("{:.2f}"), use_container_width=True)

# --- ADVANCED MULTI-INDEX COMPARISON ---
st.markdown("---")
st.subheader("🔀 Multi-Index Trend Comparison Matrix")
st.write("Compare the timeline behavior of multiple climate indicators side-by-side:")

multi_indices = st.multiselect(
    "Select indices to overlay on the chart:", 
    options=list(index_descriptions.keys()), 
    default=["TXx", "TNn", "DTR"]
)

if multi_indices:
    if selected_grid == "All Grids (Average)":
        comp_df = df.groupby('Year')[multi_indices].mean().reset_index()
    else:
        comp_df = df[df['Grid'] == selected_grid][['Year'] + multi_indices]
        
    fig_multi = px.line(
        comp_df, 
        x="Year", 
        y=multi_indices, 
        markers=True,
        labels={"value": "Value", "variable": "Index"},
        template="classic"
    )
    st.plotly_chart(fig_multi, use_container_width=True)
else:
    st.warning("Please select at least one index to display the comparison matrix chart.")