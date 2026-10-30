import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & CACHING
# ==========================================
st.set_page_config(page_title="CrisisGrid |  Triage", layout="wide")

@st.cache_resource
def load_models():
    # Loading the scaler alongside the models as required by Ridge and Lasso
    scaler = joblib.load('scaler.pkl')
    ridge = joblib.load('ridge_water.pkl')
    lasso = joblib.load('lasso_kit.pkl')
    xgb = joblib.load('xgb_severity.pkl')
    return scaler, ridge, lasso, xgb

@st.cache_data
def load_data():
    return pd.read_csv('sos_emergency_reports.csv') 

scaler, ridge_model, lasso_model, xgb_model = load_models()
df = load_data()

# ==========================================
# 2. SIDEBAR COMMAND CENTER
# ==========================================
st.sidebar.title("🚨 CrisisGrid Command")
st.sidebar.markdown("---")

st.sidebar.header("🏢 State PHED Logistics")
# We use the sidebar to visualize the limits, even though the CSV already holds the allocated data
warehouse_water = st.sidebar.slider("Central Water Reserve (L)", 100_000, 2_500_000, 500_000, step=50_000)
warehouse_kits = st.sidebar.slider("Emergency Kits", 5_000, 100_000, 25_000, step=1_000)

st.sidebar.markdown("---")
st.sidebar.header("📍 Sector Isolation")
# Using the exact column name 'Hotspot' provided in your CSV
available_hotspots = sorted(df['Hotspot'].dropna().unique().tolist())
selected_hotspot = st.sidebar.selectbox("Select K-Means Hotspot", options=["All Zones"] + available_hotspots)

if selected_hotspot != "All Zones":
    zone_data = df[df['Hotspot'] == selected_hotspot]
else:
    zone_data = df

# ==========================================
# 3. MAIN DASHBOARD UI & METRICS
# ==========================================
st.title("CrisisGrid: AI Triage & Scarcity Allocator")
st.markdown("Live emergency routing, resource allocation, and severity prediction for Jaipur.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active SOS Pings", len(zone_data))

# Calculate demand and shortages directly from your newly generated CSV columns
total_water_demand = int(zone_data['Water_Liters_Needed'].sum())
total_water_shortage = int(zone_data['water_shortage'].sum())

total_kit_demand = int(zone_data['Kits_Needed'].sum())
total_kit_shortage = int(zone_data['kit_shortage'].sum())

col2.metric("Predicted Water Demand", f"{total_water_demand:,} L")
col3.metric("Predicted Kit Demand", f"{total_kit_demand:,}")

# Display critical shortage alerts if the allocation algorithm flagged deficits
if total_water_shortage > 0:
    col4.metric("Regional Water Shortage", f"{total_water_shortage:,} L", delta="-CRITICAL", delta_color="inverse")
else:
    col4.metric("Regional Water Shortage", "0 L", delta="SUFFICIENT")

st.markdown("---")

# ==========================================
# 4. K-MEANS SPATIAL MAPPING
# ==========================================
st.subheader("📡 Unsupervised Triage Map")

if 'Latitude' in df.columns and 'Longitude' in df.columns:
    map_data = zone_data[['Latitude', 'Longitude', 'Hotspot']].copy()
    map_data.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)
   
    st.scatter_chart(
        map_data,
        x='lon',
        y='lat',
        color='Hotspot',
        height=400
    )
else:
    st.error("Missing Latitude or Longitude columns for mapping.")

# ==========================================
# 5. ALLOCATION REPORTS (DATA PREVIEW)
# ==========================================
st.subheader("📋 Live Supply Allocation Report")
st.markdown("Greedy distribution results based on AI Severity Level sorting.")
# Show the most critical columns that prove the algorithm worked
st.dataframe(zone_data[['Report_id', 'Severity_Level', 'Water_Liters_Needed', 'water_allocated', 'water_status', 'Kits_Needed', 'kit_allocated', 'kit_status']].head(15), use_container_width=True)

# ==========================================
# 6. LIVE ML INFERENCE DEMO
# ==========================================
st.markdown("---")
st.subheader("🧠 Live AI Inference")
st.markdown("Adjust environmental factors to run real-time predictions through Ridge, Lasso, and XGBoost.")

demo_col1, demo_col2, demo_col3 = st.columns(3)
sim_trapped = demo_col1.slider("Total People Trapped", 0, 500, 150)
# Matched to your exact spelling "Temperatur_C"
sim_temp = demo_col2.slider("Temperature (°C)", 20, 50, 35)
sim_damage = demo_col3.slider("Infra Damage (%)", 0, 100, 45)

if st.button("Run Predictive Models"):
    # 1. Filter out all non-numeric, target, and text columns to perfectly match the 8 training features
    cols_to_drop = [
        'Report_id', 'Zone', 'Water_Liters_Needed', 'Kits_Needed', 
        'Severity_Level', 'water_status', 'kit_status', 
        'water_allocated', 'kit_allocated', 'water_shortage', 'kit_shortage'
    ]
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # 2. Create a base vector using dataset averages to prevent Feature Mismatch Errors
    input_vector = numeric_df.mean().to_frame().T
    
    # 3. Inject the UI slider values
    if 'Total_Trapped' in input_vector.columns:
        input_vector['Total_Trapped'] = sim_trapped
    if 'Temperatur_C' in input_vector.columns:
        input_vector['Temperatur_C'] = sim_temp
    if 'Infra_Damage_Pct' in input_vector.columns:
        input_vector['Infra_Damage_Pct'] = sim_damage

    # 4. Predict Water & Kits (Requires scaling)
    scaled_input = scaler.transform(input_vector)
    water_pred = ridge_model.predict(scaled_input)[0]
    kit_pred = lasso_model.predict(scaled_input)[0]
    
    # 5. Predict Severity (Directly from inputs)
    sev_pred = xgb_model.predict(input_vector)[0]
    
    # 6. Display Outputs
    res1, res2, res3 = st.columns(3)
    res1.success(f"💧 Ridge Water Output: {int(water_pred)} L")
    res2.info(f"🚑 Lasso Kit Output: {int(kit_pred)} Kits")
    # If XGBoost output is 0-4, we add 1 for display purposes (1-5 scale)
    res3.error(f"🌲 XGBoost Severity: Level {int(sev_pred) + 1} / 5")