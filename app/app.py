import streamlit as st
import pandas as pd
import numpy as np
import folium

from pathlib import Path
from streamlit_folium import st_folium

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Ride-Sharing Optimizer",
    page_icon="🚖",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent

CLEAN = ROOT / "data" / "processed" / "cleaned_trips.csv"
DEMAND = ROOT / "data" / "processed" / "demand_labels.csv"
HOTSPOTS = ROOT / "data" / "processed" / "hotspot_clusters.csv"
RAW = ROOT / "data" / "raw" / "yellow_tripdata_sample.csv"


# =========================================================
# 2. COLUMN CONFIGURATION
# =========================================================

LAT, LON = "pickup_latitude", "pickup_longitude"
DIST, PAX = "trip_distance", "passenger_count"
HOUR, DOW = "hour", "day_of_week"
FARE, DUR = "fare_amount", "duration_min"
CLUSTER, SURGE = "cluster", "high_demand"

FARE_FEATS = [DIST, PAX, HOUR, DOW]
ETA_FEATS = [DIST, HOUR, DOW, LAT, LON]
SURGE_FEATS = [HOUR, CLUSTER]

DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

COLORS = [
    "red", "blue", "green", "purple",
    "orange", "darkred", "cadetblue", "black"
]


# =========================================================
# 3. DATA LOADING
# =========================================================

@st.cache_data
def load(path, n=None):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=42)

    return df


def clean_numeric_columns(df, columns):
    """Convert required columns to numeric and remove missing values."""
    df = df.copy()

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=columns)

    return df


# =========================================================
# 4. GEOGRAPHIC HELPERS
# =========================================================

def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance between two coordinates."""
    radius = 3958.8

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(p1)
        * np.cos(p2)
        * np.sin(dlmb / 2) ** 2
    )

    return float(
        2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


def nearest_cluster(lat, lon, centroids):
    distances = (
        (centroids[LAT] - lat) ** 2
        + (centroids[LON] - lon) ** 2
    )

    return int(distances.idxmin())


def assign_nearest_clusters(demand, centroids):
    """Assign each demand cell to its nearest hotspot centroid."""
    coords = demand[
        ["pickup_cell_lat", "pickup_cell_lon"]
    ].to_numpy()

    center_coords = centroids[
        [LAT, LON]
    ].to_numpy()

    distances = (
        (coords[:, None, 0] - center_coords[None, :, 0]) ** 2
        + (coords[:, None, 1] - center_coords[None, :, 1]) ** 2
    )

    demand = demand.copy()

    demand[CLUSTER] = centroids.index.to_numpy()[
        distances.argmin(axis=1)
    ]

    return demand


# =========================================================
# 5. TRAIN PREDICTION MODELS
# =========================================================

@st.cache_resource
def train_models():
    # ---------------- FARE AND ETA DATA ----------------
    df = load(CLEAN, 100000)

    required = list(set(
        FARE_FEATS + ETA_FEATS + [FARE, DUR]
    ))

    df = clean_numeric_columns(df, required)

    if len(df) < 20:
        raise ValueError(
            "Not enough valid cleaned trip records to train models."
        )

    # Fare: Linear Regression
    fare_lr = LinearRegression()
    fare_lr.fit(df[FARE_FEATS], df[FARE])

    # Fare: Random Forest
    fare_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )
    fare_rf.fit(df[FARE_FEATS], df[FARE])

    # ETA: k-NN
    eta = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=11)
    )
    eta.fit(df[ETA_FEATS], df[DUR])

    # ---------------- HOTSPOT CENTROIDS ----------------
    hs = load(HOTSPOTS)

    hs = clean_numeric_columns(
        hs, [CLUSTER, LAT, LON]
    )

    centroids = hs.groupby(CLUSTER)[[LAT, LON]].mean()

    if centroids.empty:
        raise ValueError("No valid hotspot centroids were found.")

    # ---------------- SURGE CLASSIFIER ----------------
    demand = load(DEMAND)

    demand = clean_numeric_columns(
        demand,
        [HOUR, "pickup_cell_lat", "pickup_cell_lon", SURGE]
    )

    demand = assign_nearest_clusters(demand, centroids)

    demand = clean_numeric_columns(
        demand, SURGE_FEATS + [SURGE]
    )

    if demand[SURGE].nunique() < 2:
        raise ValueError(
            "Surge training data must contain both demand classes."
        )

    surge = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    surge.fit(demand[SURGE_FEATS], demand[SURGE])

    return fare_lr, fare_rf, eta, surge, centroids


# =========================================================
# 6. MODEL EVALUATION FUNCTIONS
# =========================================================

@st.cache_data
def evaluate_regression_models():
    """Compare regression models on a held-out test set."""

    df = load(CLEAN, 100000)

    required = list(set(
        FARE_FEATS + ETA_FEATS + [FARE, DUR]
    ))

    df = clean_numeric_columns(df, required)

    if len(df) < 20:
        raise ValueError(
            "Not enough valid rows for regression evaluation."
        )

    results = []

    # -----------------------------------------------------
    # FARE PREDICTION
    # -----------------------------------------------------

    X = df[FARE_FEATS]
    y = df[FARE]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    fare_models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            n_jobs=-1,
            random_state=42
        )
    }

    for name, model in fare_models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results.append({
            "Task": "Fare Prediction",
            "Model": name,
            "MAE": mean_absolute_error(
                y_test, predictions
            ),
            "RMSE": np.sqrt(
                mean_squared_error(y_test, predictions)
            ),
            "R2 Score": r2_score(
                y_test, predictions
            )
        })

    # -----------------------------------------------------
    # ETA PREDICTION
    # -----------------------------------------------------

    X = df[ETA_FEATS]
    y = df[DUR]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    eta_models = {
        "k-NN": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=11)
        ),

        "Linear Regression": make_pipeline(
            StandardScaler(),
            LinearRegression()
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            n_jobs=-1,
            random_state=42
        )
    }

    for name, model in eta_models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results.append({
            "Task": "ETA Prediction",
            "Model": name,
            "MAE": mean_absolute_error(
                y_test, predictions
            ),
            "RMSE": np.sqrt(
                mean_squared_error(y_test, predictions)
            ),
            "R2 Score": r2_score(
                y_test, predictions
            )
        })

    return pd.DataFrame(results)


@st.cache_data
def evaluate_surge_model():
    """Evaluate surge classification on a held-out test set."""

    demand = load(DEMAND)
    hs = load(HOTSPOTS)

    demand = clean_numeric_columns(
        demand,
        [HOUR, "pickup_cell_lat", "pickup_cell_lon", SURGE]
    )

    hs = clean_numeric_columns(
        hs, [CLUSTER, LAT, LON]
    )

    centroids = hs.groupby(CLUSTER)[[LAT, LON]].mean()

    if centroids.empty:
        raise ValueError("No valid hotspot centroids found.")

    demand = assign_nearest_clusters(demand, centroids)

    demand = clean_numeric_columns(
        demand, SURGE_FEATS + [SURGE]
    )

    # Normalize target labels to 0 and 1.
    # Assumes the target represents binary high-demand labels.
    labels = set(demand[SURGE].unique())

    if labels.issubset({0, 1, 0.0, 1.0, False, True}):
        demand[SURGE] = demand[SURGE].astype(int)
    else:
        raise ValueError(
            "The high_demand column must be encoded as 0 and 1."
        )

    X = demand[SURGE_FEATS]
    y = demand[SURGE]

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError(
            "Both surge classes need at least two samples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = pd.DataFrame([{
        "Model": "Random Forest Classifier",
        "Accuracy": accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    }])

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    return metrics, cm


# =========================================================
# 7. APP HEADER
# =========================================================

st.title("🚖 Smart Ride-Sharing Demand & Pricing Optimizer")

st.caption(
    "Predict fare, ETA and surge status, "
    "and explore demand hotspots across NYC."
)


# =========================================================
# 8. LOAD MODELS
# =========================================================

try:
    (
        fare_lr,
        fare_rf,
        eta_model,
        surge_model,
        centroids
    ) = train_models()

except Exception as e:
    st.error(
        f"Could not load data or train models: {e}"
    )
    st.stop()


# =========================================================
# 9. APPLICATION TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Predict",
    "Hotspot Map",
    "Model Comparison",
    "Data Explorer"
])


# =========================================================
# TAB 1: PREDICTION
# =========================================================

with tab1:
    st.subheader("Trip details")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Pickup**")

        p_lat = st.number_input(
            "Pickup latitude",
            value=40.7580,
            format="%.4f"
        )

        p_lon = st.number_input(
            "Pickup longitude",
            value=-73.9855,
            format="%.4f"
        )

    with c2:
        st.markdown("**Drop-off**")

        d_lat = st.number_input(
            "Drop-off latitude",
            value=40.6413,
            format="%.4f"
        )

        d_lon = st.number_input(
            "Drop-off longitude",
            value=-73.7781,
            format="%.4f"
        )

    c3, c4, c5 = st.columns(3)

    hour = c3.slider(
        "Pickup hour", 0, 23, 18
    )

    day = c4.selectbox(
        "Day of week", DAYS, index=4
    )

    pax = c5.number_input(
        "Passengers",
        min_value=1,
        max_value=6,
        value=1
    )

    if st.button(
        "Predict",
        type="primary",
        use_container_width=True
    ):
        dow = DAYS.index(day)

        dist = haversine_miles(
            p_lat, p_lon, d_lat, d_lon
        )

        cl = nearest_cluster(
            p_lat, p_lon, centroids
        )

        fare_in = pd.DataFrame(
            [[dist, pax, hour, dow]],
            columns=FARE_FEATS
        )

        eta_in = pd.DataFrame(
            [[dist, hour, dow, p_lat, p_lon]],
            columns=ETA_FEATS
        )

        surge_in = pd.DataFrame(
            [[hour, cl]],
            columns=SURGE_FEATS
        )

        fare_rf_pred = fare_rf.predict(
            fare_in
        )[0]

        fare_lr_pred = fare_lr.predict(
            fare_in
        )[0]

        eta_min = eta_model.predict(
            eta_in
        )[0]

        # Use the actual class-1 probability rather than
        # assuming probability column 1 always means high demand.
        classes = list(surge_model.classes_)
        high_index = classes.index(1)

        surge_prob = surge_model.predict_proba(
            surge_in
        )[0][high_index]

        is_high = surge_prob >= 0.5

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Fare (Random Forest)",
            f"${fare_rf_pred:,.2f}"
        )

        m2.metric(
            "Fare (Linear Regression)",
            f"${fare_lr_pred:,.2f}"
        )

        m3.metric(
            "ETA (k-NN)",
            f"{eta_min:,.0f} min"
        )

        m4.metric(
            "Distance",
            f"{dist:,.1f} mi"
        )

        if is_high:
            st.error(
                f"High-demand zone (hotspot {cl}) — "
                f"surge probability {surge_prob:.0%}"
            )
        else:
            st.success(
                f"Normal demand (hotspot {cl}) — "
                f"surge probability {surge_prob:.0%}"
            )

        map_obj = folium.Map(
            location=[p_lat, p_lon],
            zoom_start=11
        )

        folium.Marker(
            [p_lat, p_lon],
            tooltip="Pickup",
            icon=folium.Icon(color="green")
        ).add_to(map_obj)

        folium.Marker(
            [d_lat, d_lon],
            tooltip="Drop-off",
            icon=folium.Icon(color="red")
        ).add_to(map_obj)

        folium.PolyLine(
            [[p_lat, p_lon], [d_lat, d_lon]],
            color="blue",
            weight=3
        ).add_to(map_obj)

        st_folium(
            map_obj,
            height=420,
            use_container_width=True,
            returned_objects=[]
        )

        st.caption(
            "The map displays a straight line between pickup "
            "and drop-off, not a road-routed driving path."
        )


# =========================================================
# TAB 2: HOTSPOT MAP
# =========================================================

with tab2:
    st.subheader("Demand hotspots (K-Means clusters)")

    try:
        hs = load(HOTSPOTS)

        hs = clean_numeric_columns(
            hs, [CLUSTER, LAT, LON]
        )

        clusters = sorted(
            hs[CLUSTER].unique()
        )

        chosen = st.multiselect(
            "Clusters to show",
            clusters,
            default=clusters
        )

        sample_n = st.slider(
            "Points to plot",
            min_value=500,
            max_value=10000,
            value=3000,
            step=500
        )

        view = hs[
            hs[CLUSTER].isin(chosen)
        ]

        if view.empty:
            st.warning(
                "Select at least one cluster to display."
            )
        else:
            view = view.sample(
                n=min(sample_n, len(view)),
                random_state=1
            )

            map_obj = folium.Map(
                location=[
                    view[LAT].mean(),
                    view[LON].mean()
                ],
                zoom_start=11,
                tiles="cartodbpositron"
            )

            for _, row in view.iterrows():
                folium.CircleMarker(
                    [row[LAT], row[LON]],
                    radius=2,
                    color=COLORS[
                        int(row[CLUSTER]) % len(COLORS)
                    ],
                    fill=True,
                    fill_opacity=0.6
                ).add_to(map_obj)

            valid_centroids = [
                c for c in chosen
                if c in centroids.index
            ]

            for cid, row in centroids.loc[
                valid_centroids
            ].iterrows():
                folium.Marker(
                    [row[LAT], row[LON]],
                    tooltip=f"Hotspot {cid}",
                    icon=folium.Icon(
                        color=COLORS[
                            int(cid) % len(COLORS)
                        ],
                        icon="star"
                    )
                ).add_to(map_obj)

            st_folium(
                map_obj,
                height=520,
                use_container_width=True,
                returned_objects=[]
            )

        st.markdown("**Trips per cluster**")

        st.bar_chart(
            hs[CLUSTER].value_counts().sort_index()
        )

    except Exception as e:
        st.error(
            f"Could not display hotspot map: {e}"
        )


# =========================================================
# TAB 3: MODEL COMPARISON
# =========================================================

with tab3:
    st.subheader("📊 Model Comparison Dashboard")

    st.markdown(
        """
        Compare fare and travel-time prediction models using
        held-out test data. Evaluate surge classification using
        accuracy, precision, recall, F1 score and a confusion matrix.
        """
    )

    # -----------------------------------------------------
    # REGRESSION COMPARISON
    # -----------------------------------------------------

    st.markdown("### 1. Fare and ETA prediction")

    try:
        regression_results = evaluate_regression_models()

        st.dataframe(
            regression_results.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "MAE and RMSE: lower is better. "
            "R²: higher is better. "
            "Evaluation uses an 80/20 train/test split."
        )

        selected_task = st.selectbox(
            "Select prediction task",
            ["Fare Prediction", "ETA Prediction"],
            key="comparison_task"
        )

        selected = regression_results[
            regression_results["Task"] == selected_task
        ].set_index("Model")

        chart_metric = st.selectbox(
            "Select comparison metric",
            ["RMSE", "MAE", "R2 Score"],
            key="comparison_metric"
        )

        st.markdown(
            f"#### {selected_task}: {chart_metric}"
        )

        st.bar_chart(
            selected[[chart_metric]]
        )

        if chart_metric == "R2 Score":
            best_model = selected[chart_metric].idxmax()
            best_value = selected[chart_metric].max()
        else:
            best_model = selected[chart_metric].idxmin()
            best_value = selected[chart_metric].min()

        st.success(
            f"Best {selected_task.lower()} model by "
            f"{chart_metric}: {best_model} "
            f"({chart_metric} = {best_value:.4f})"
        )

    except Exception as e:
        st.error(
            f"Regression model evaluation failed: {e}"
        )

    st.divider()

    # -----------------------------------------------------
    # SURGE CLASSIFICATION COMPARISON
    # -----------------------------------------------------

    st.markdown("### 2. Surge classification")

    try:
        surge_metrics, cm = evaluate_surge_model()

        st.dataframe(
            surge_metrics.round(4),
            use_container_width=True,
            hide_index=True
        )

        row = surge_metrics.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{row['Accuracy']:.3f}"
        )

        c2.metric(
            "Precision",
            f"{row['Precision']:.3f}"
        )

        c3.metric(
            "Recall",
            f"{row['Recall']:.3f}"
        )

        c4.metric(
            "F1 Score",
            f"{row['F1 Score']:.3f}"
        )

        st.markdown("#### Confusion Matrix")

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual Normal",
                "Actual High Demand"
            ],
            columns=[
                "Predicted Normal",
                "Predicted High Demand"
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        st.caption(
            "Class 1 represents high demand. "
            "Precision measures the reliability of high-demand "
            "predictions; recall measures how many actual "
            "high-demand cases were detected."
        )

        st.bar_chart(
            surge_metrics.set_index("Model")[
                ["Accuracy", "Precision", "Recall", "F1 Score"]
            ].T
        )

    except Exception as e:
        st.error(
            f"Surge model evaluation failed: {e}"
        )

    st.divider()

    # -----------------------------------------------------
    # SAVED REPORTS
    # -----------------------------------------------------

    st.markdown("### 3. Saved experiment reports")

    SUMMARY = ROOT / "reports" / "summary_best_models.csv"
    COMPARISON = ROOT / "reports" / "model_comparison.csv"
    FIGS = ROOT / "reports" / "figures"

    for label, path in [
        ("Best model per task", SUMMARY),
        ("Saved classification metrics", COMPARISON)
    ]:
        if path.exists():
            st.markdown(f"**{label}**")

            st.dataframe(
                pd.read_csv(path),
                use_container_width=True,
                hide_index=True
            )

    figs = (
        sorted(FIGS.glob("*.png"))
        if FIGS.exists()
        else []
    )

    if figs:
        st.markdown("### Saved evaluation figures")

        cols = st.columns(2)

        for i, fig in enumerate(figs):
            cols[i % 2].image(
                str(fig),
                caption=fig.stem.replace(
                    "_", " "
                ).title(),
                use_container_width=True
            )

    if not figs and not SUMMARY.exists() and not COMPARISON.exists():
        st.info(
            "No saved experiment reports found. "
            "Live evaluation results above are calculated "
            "from your datasets."
        )


# =========================================================
# TAB 4: DATA EXPLORER
# =========================================================

with tab4:
    st.subheader("Data Explorer")

    files = {
        "Raw sample": RAW,
        "Cleaned trips": CLEAN,
        "Demand labels": DEMAND,
        "Hotspot clusters": HOTSPOTS
    }

    pick = st.selectbox(
        "Dataset",
        list(files.keys())
    )

    try:
        data = load(files[pick])

        a, b = st.columns(2)

        a.metric(
            "Rows",
            f"{len(data):,}"
        )

        b.metric(
            "Columns",
            data.shape[1]
        )

        st.markdown("**Dataset preview**")

        st.dataframe(
            data.head(200),
            use_container_width=True
        )

        st.markdown("**Summary statistics**")

        st.dataframe(
            data.describe(),
            use_container_width=True
        )

        st.markdown("**Missing values per column**")

        missing = data.isnull().sum()

        st.dataframe(
            missing.rename("Missing Values").to_frame(),
            use_container_width=True
        )

    except Exception as e:
        st.error(
            f"Could not load selected dataset: {e}"
        )
