import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = Path("synthetic_landslide_data.csv")
MODEL_PATH = Path("landslide_model.pkl")
RANDOM_STATE = 42
FEATURES = [
    "soil_moisture_surface",
    "soil_moisture_mid",
    "rainfall",
    "tilt_angle",
    "vibration",
]
LOCATIONS = [
    "North Ridge",
    "Valley Gate",
    "Hilltop East",
    "River Bend",
    "Station Alpha",
]


def set_page_style() -> None:
    st.set_page_config(page_title="Landslide AI Monitor", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {background-color: white; color: #1f2937;}
        [data-testid="stSidebar"] {background-color: #f9fafb;}
        .card {
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 16px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .alert-high {
            border-left: 5px solid #dc2626;
            padding: 12px;
            background: #fef2f2;
            color: #991b1b;
            border-radius: 8px;
            font-weight: 600;
        }
        .alert-safe {
            border-left: 5px solid #16a34a;
            padding: 12px;
            background: #f0fdf4;
            color: #166534;
            border-radius: 8px;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_synthetic_dataset(rows: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    soil_surface = rng.uniform(5, 95, rows)
    soil_mid = np.clip(soil_surface + rng.normal(0, 12, rows), 4, 98)
    rainfall = rng.gamma(shape=2.0, scale=20.0, size=rows)
    tilt = np.clip(rng.normal(8, 5.5, rows), 0, 45)
    vibration = np.clip(rng.normal(2.5, 1.3, rows), 0, 12)

    risk_signal = (
        0.28 * (soil_surface / 100)
        + 0.25 * (soil_mid / 100)
        + 0.23 * np.clip(rainfall / 150, 0, 1)
        + 0.16 * np.clip(tilt / 30, 0, 1)
        + 0.08 * np.clip(vibration / 10, 0, 1)
    )
    noise = rng.normal(0, 0.05, rows)
    landslide_event = ((risk_signal + noise) > 0.58).astype(int)

    df = pd.DataFrame(
        {
            "soil_moisture_surface": soil_surface,
            "soil_moisture_mid": soil_mid,
            "rainfall": rainfall,
            "tilt_angle": tilt,
            "vibration": vibration,
            "landslide": landslide_event,
        }
    )
    return df


def load_or_create_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    df = generate_synthetic_dataset()
    df.to_csv(DATA_PATH, index=False)
    return df


def train_and_save_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, float, np.ndarray, pd.DataFrame, pd.Series]:
    X = df[FEATURES]
    y = df["landslide"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(n_estimators=220, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    return model, accuracy, cm, X_test, y_test


def load_or_train_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, float, np.ndarray, pd.DataFrame, pd.Series]:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        X = df[FEATURES]
        y = df["landslide"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        y_pred = model.predict(X_test)
        return model, accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred), X_test, y_test
    return train_and_save_model(df)


def classify_risk(probability: float) -> str:
    if probability < 0.4:
        return "Low"
    if probability < 0.7:
        return "Medium"
    return "High"


def simulate_sensor_input(location: str) -> Dict[str, float]:
    seed = int(time.time()) + sum(ord(ch) for ch in location)
    rng = np.random.default_rng(seed)
    return {
        "soil_moisture_surface": float(np.clip(rng.normal(55, 18), 5, 98)),
        "soil_moisture_mid": float(np.clip(rng.normal(50, 16), 4, 98)),
        "rainfall": float(np.clip(rng.gamma(2.2, 17), 0, 180)),
        "tilt_angle": float(np.clip(rng.normal(10, 5), 0, 45)),
        "vibration": float(np.clip(rng.normal(2.8, 1.4), 0, 12)),
    }


def render_login() -> None:
    st.title("Landslide Prediction Dashboard")
    st.caption("Secure access for monitoring teams")

    with st.container(border=True):
        st.subheader("Login")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        location = st.selectbox("Monitoring Location", LOCATIONS)

        if st.button("Sign in", use_container_width=True):
            if username.strip().lower() == "admin" and password == "terraguard123":
                st.session_state.authenticated = True
                st.session_state.location = location
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials. Try admin / terraguard123")


def ensure_history() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def push_sensor_reading(reading: Dict[str, float], probability: float) -> None:
    entry = {
        "timestamp": pd.Timestamp.now(),
        **reading,
        "risk_probability": probability,
    }
    st.session_state.history.append(entry)
    st.session_state.history = st.session_state.history[-60:]


def risk_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"valueformat": ".2f", "suffix": ""},
            title={"text": "Predicted Risk Probability"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#111827"},
                "steps": [
                    {"range": [0, 0.4], "color": "#bbf7d0"},
                    {"range": [0.4, 0.7], "color": "#fef08a"},
                    {"range": [0.7, 1.0], "color": "#fecaca"},
                ],
                "threshold": {"line": {"color": "#dc2626", "width": 3}, "thickness": 0.8, "value": 0.7},
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="white")
    return fig


def render_dashboard(model: RandomForestClassifier, accuracy: float, cm: np.ndarray) -> None:
    st.title("AI-Based Real-Time Landslide Prediction")
    st.caption(f"Location: {st.session_state.location}")

    top_a, top_b, top_c = st.columns([1, 1, 1])
    with top_a:
        auto_refresh = st.toggle("Auto refresh every 3 sec", value=True)
    with top_b:
        manual = st.button("Simulate next reading")
    with top_c:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.history = []
            st.rerun()

    reading = simulate_sensor_input(st.session_state.location)
    proba = float(model.predict_proba(pd.DataFrame([reading]))[0][1])
    risk_level = classify_risk(proba)

    if manual or auto_refresh:
        push_sensor_reading(reading, proba)

    history_df = pd.DataFrame(st.session_state.history)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Live Sensor Values")
        sensor_df = pd.DataFrame([reading]).T.reset_index()
        sensor_df.columns = ["Sensor", "Value"]
        st.dataframe(sensor_df, use_container_width=True, hide_index=True)

        st.subheader("Alert")
        if risk_level == "High":
            st.markdown(
                f'<div class="alert-high">⚠️ High Risk detected at {st.session_state.location}. Immediate inspection recommended.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alert-safe">System stable. Current risk category: {risk_level}.</div>',
                unsafe_allow_html=True,
            )

    with c2:
        st.plotly_chart(risk_gauge(proba), use_container_width=True)
        st.metric("Risk Category", risk_level)
        st.metric("Model Accuracy", f"{accuracy:.3f}")

    if not history_df.empty:
        st.subheader("Time-Series Sensor and Risk Trend")
        chart_df = history_df.set_index("timestamp")
        st.line_chart(chart_df[["risk_probability", "rainfall", "tilt_angle", "vibration"]], height=260)

    st.subheader("Model Performance Metrics")
    perf_col1, perf_col2 = st.columns([1, 1])
    with perf_col1:
        st.write("Confusion Matrix")
        cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"])
        st.dataframe(cm_df, use_container_width=True)
    with perf_col2:
        st.write("Prediction Confidence")
        st.write(f"Probability: **{proba:.3f}**")
        st.write("Thresholds: Low < 0.4, Medium 0.4-0.7, High > 0.7")

    if auto_refresh:
        time.sleep(3)
        st.rerun()


def main() -> None:
    set_page_style()
    ensure_history()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    dataset = load_or_create_dataset()
    model, accuracy, cm, _, _ = load_or_train_model(dataset)

    if not st.session_state.authenticated:
        render_login()
    else:
        render_dashboard(model, accuracy, cm)


if __name__ == "__main__":
    main()
