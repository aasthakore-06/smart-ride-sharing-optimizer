import streamlit as st
import pandas as pd
import numpy as np
import folium
from pathlib import Path
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Smart Ride-Sharing Optimizer", page_icon="🚖", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
CLEAN = str(ROOT / "data/processed/cleaned_trips.csv")
DEMAND = str(ROOT / "data/processed/demand_labels.csv")
HOTSPOTS = str(ROOT / "data/processed/hotspot_clusters.csv")
RAW = str(ROOT / "data/raw/yellow_tripdata_sample.csv")

LAT, LON = "pickup_latitude", "pickup_longitude"
DIST, PAX = "trip_distance", "passenger_count"
HOUR, DOW = "hour", "day_of_week"
FARE, DUR = "fare_amount", "duration_min"
CLUSTER, SURGE = "cluster", "high_demand"

FARE_FEATS = [DIST, PAX, HOUR, DOW]
ETA_FEATS = [DIST, HOUR, DOW, LAT, LON]
SURGE_FEATS = [HOUR, CLUSTER]

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
COLORS = ["red","blue","green","purple","orange","darkred","cadetblue","black"]

@st.cache_data
def load(path, n=None):
    df = pd.read_csv(path)
    if n and len(df) > n:
        df = df.sample(n, random_state=42)
    return df

@st.cache_resource
def train_models():
    df = load(CLEAN, 100000)
    
    fare_lr = LinearRegression().fit(df[FARE_FEATS], df[FARE])
    fare_rf = RandomForestRegressor(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
    ).fit(df[FARE_FEATS], df[FARE])
    
    eta = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=11)
    ).fit(df[ETA_FEATS], df[DUR])
    
    hs = load(HOTSPOTS)
    centroids = hs.groupby(CLUSTER)[[LAT, LON]].mean()
    
    demand = load(DEMAND)
    def assign_cluster(row):
        d = (centroids[LAT] - row["pickup_cell_lat"])**2 + \
            (centroids[LON] - row["pickup_cell_lon"])**2
        return int(d.idxmin())
    
    demand[CLUSTER] = demand.apply(assign_cluster, axis=1)
    
    surge = RandomForestClassifier(
        n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=42
    ).fit(demand[SURGE_FEATS], demand[SURGE])
    
    return fare_lr, fare_rf, eta, surge, centroids

def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))

def nearest_cluster(lat, lon, centroids):
    d = (centroids[LAT] - lat)**2 + (centroids[LON] - lon)**2
    return int(d.idxmin())

st.title("🚖 Smart Ride-Sharing Demand & Pricing Optimizer")
st.caption("Predict fare, ETA and explore demand hotspots")

try:
    fare_lr, fare_rf, eta_model, surge_model, centroids = train_models()
except Exception as e:
    st.error(f"Could not load data or train models: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Hotspot Map", "Model Comparison", "Data Explorer"])

with tab1:
    st.subheader("Trip details")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Pickup**")
        p_lat = st.number_input("Pickup latitude", value=40.7580, format="%.4f")
        p_lon = st.number_input("Pickup longitude", value=-73.9855, format="%.4f")
    with c2:
        st.markdown("**Drop-off**")
        d_lat = st.number_input("Drop-off latitude", value=40.6413, format="%.4f")
        d_lon = st.number_input("Drop-off longitude", value=-73.7781, format="%.4f")
        
    c3, c4, c5 = st.columns(3)
    hour = c3.slider("Pickup hour", 0, 23, 18)
    day = c4.selectbox("Day of week", DAYS, index=4)
    pax = c5.number_input("Passengers", min_value=1, max_value=6, value=1)
    
    if st.button("Predict", type="primary", use_container_width=True):
        dow = DAYS.index(day)
        dist = float(haversine_miles(p_lat, p_lon, d_lat, d_lon))
        cl = nearest_cluster(p_lat, p_lon, centroids)
        
        fare_in = pd.DataFrame([[dist, pax, hour, dow]], columns=FARE_FEATS)
        eta_in = pd.DataFrame([[dist, hour, dow, p_lat, p_lon]], columns=ETA_FEATS)
        surge_in = pd.DataFrame([[hour, cl]], columns=SURGE_FEATS)
        
        fare_rf_pred = fare_rf.predict(fare_in)[0]
        fare_lr_pred = fare_lr.predict(fare_in)[0]
        eta_min = eta_model.predict(eta_in)[0]
        surge_prob = surge_model.predict_proba(surge_in)[0][1]
        is_high = surge_prob >= 0.5
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fare (Random Forest)", f"${fare_rf_pred:,.2f}")
        m2.metric("Fare (Linear Reg.)", f"${fare_lr_pred:,.2f}")
        m3.metric("ETA (k-NN)", f"{eta_min:,.0f} min")
        m4.metric("Distance", f"{dist:,.1f} mi")
        
        if is_high:
            st.error(f"High-demand zone (hotspot {cl}) — surge probability {surge_prob:.0%}")
        else:
            st.success(f"Normal demand (hotspot {cl}) — surge probability {surge_prob:.0%}")
            
        m = folium.Map(location=[p_lat, p_lon], zoom_start=11)
        folium.Marker([p_lat, p_lon], tooltip="Pickup", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker([d_lat, d_lon], tooltip="Drop-off", icon=folium.Icon(color="red")).add_to(m)
        folium.PolyLine([[p_lat, p_lon],[d_lat, d_lon]], color="blue", weight=3).add_to(m)
        st_folium(m, height=420, use_container_width=True, returned_objects=[])

with tab2:
    st.subheader("Demand hotspots (K-Means clusters)")
    hs = load(HOTSPOTS)
    clusters = sorted(hs[CLUSTER].unique())
    chosen = st.multiselect("Clusters to show", clusters, default=clusters)
    sample_n = st.slider("Points to plot", 500, 10000, 3000, step=500)
    
    view = hs[hs[CLUSTER].isin(chosen)]
    view = view.sample(min(sample_n, len(view)), random_state=1)
    
    m = folium.Map(location=[view[LAT].mean(), view[LON].mean()], zoom_start=11, tiles="cartodbpositron")
    for _, r in view.iterrows():
        folium.CircleMarker(
            [r[LAT], r[LON]], radius=2, color=COLORS[int(r[CLUSTER]) % len(COLORS)], fill=True, fill_opacity=0.6
        ).add_to(m)
        
    for cid, r in centroids.loc[[c for c in chosen if c in centroids.index]].iterrows():
        folium.Marker(
            [r[LAT], r[LON]], tooltip=f"Hotspot {cid}", icon=folium.Icon(color=COLORS[int(cid) % len(COLORS)], icon="star")
        ).add_to(m)
        
    st_folium(m, height=520, use_container_width=True, returned_objects=[])
    st.markdown("**Trips per cluster**")
    st.bar_chart(hs[CLUSTER].value_counts().sort_index())

with tab3:
    st.subheader("Model comparison")
    SUMMARY = ROOT / "reports/summary_best_models.csv"
    COMPARISON = ROOT / "reports/model_comparison.csv"
    FIGS = ROOT / "reports/figures"
    
    has_files = SUMMARY.exists() or (FIGS.exists() and len(list(FIGS.glob("*.png"))) > 0)
    
    if has_files:
        if SUMMARY.exists():
            st.markdown("**Best model per task**")
            st.dataframe(pd.read_csv(SUMMARY), use_container_width=True, hide_index=True)
        if COMPARISON.exists():
            st.markdown("**Surge classification metrics**")
            st.dataframe(pd.read_csv(COMPARISON), use_container_width=True, hide_index=True)
            
        figs = sorted(FIGS.glob("*.png")) if FIGS.exists() else []
        if figs:
            st.markdown("**Figures**")
            cols = st.columns(2)
            for i, f in enumerate(figs):
                cols[i % 2].image(str(f), caption=f.stem.replace("_"," ").title(), use_container_width=True)
    else:
        st.warning("Reports directory missing or empty. Calculating validation statistics live...")
        
        df_clean = load(CLEAN, 20000)
        df_demand = load(DEMAND, 20000)
        
        lr_pred = fare_lr.predict(df_clean[FARE_FEATS])
        rf_pred = fare_rf.predict(df_clean[FARE_FEATS])
        
        summary_data = {
            "Task": ["Fare Prediction (Linear Reg.)", "Fare Prediction (Random Forest)"],
            "MAE": [f"${mean_absolute_error(df_clean[FARE], lr_pred):.2f}", f"${mean_absolute_error(df_clean[FARE], rf_pred):.2f}"],
            "R2 Score": [f"{r2_score(df_clean[FARE], lr_pred):.4f}", f"{r2_score(df_clean[FARE], rf_pred):.4f}"]
        }
        st.markdown("**Best model per task (Live Validation)**")
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        def assign_cluster_val(row):
            d = (centroids[LAT] - row["pickup_cell_lat"])**2 + (centroids[LON] - row["pickup_cell_lon"])**2
            return int(d.idxmin())
        df_demand[CLUSTER] = df_demand.apply(assign_cluster_val, axis=1)
        
        s_pred = surge_model.predict(df_demand[SURGE_FEATS])
        comp_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Value": [
                f"{accuracy_score(df_demand[SURGE], s_pred):.4f}",
                f"{precision_score(df_demand[SURGE], s_pred, zero_division=0):.4f}",
                f"{recall_score(df_demand[SURGE], s_pred, zero_division=0):.4f}",
                f"{f1_score(df_demand[SURGE], s_pred, zero_division=0):.4f}"
            ]
        }
        st.markdown("**Surge classification metrics (Live Validation)**")
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
        
        st.markdown("**Feature Importances (Surge Model)**")
        feat_imp_df = pd.DataFrame({
            "Feature": SURGE_FEATS,
            "Importance": surge_model.feature_importances_
        }).set_index("Feature")
        st.bar_chart(feat_imp_df)

with tab4:
    st.subheader("Data explorer")
