import os
import glob

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt


st.set_page_config(page_title="PC Stroke Outcome Prediction", layout="centered")
st.title("Posterior Circulation Stroke 90-day Functional Outcome Prediction in EVT Treated Patients")
st.write("As seen in [Publication Pending...]")
st.write("Unfavourable functional outcome defined as mRS >2 at 90-days")
st.write("This model is only suitable for patients eligible for EVT")


MAIN_MODEL_PATH = {
    "Baseline": "baseline_model.pkl",
    "M3": "m3_model.pkl",
}

BOOT_DIR = {
    "Baseline": "baseline_boots",
    "M3": "m3_boots",
}

FEATURES = {
    "Baseline": ["age", "admission_NIHSS", "IVT", "sex"],
    "M3": ["age", "admission_NIHSS", "IVT", "sex", "day_NIHSS"],
}

PLOT_CSV_PATH = {
    "Baseline": "baseline_predictions.csv",
    "M3": "m3_predictions.csv",
}

PRED_COLS = {
    "main": "Predicted Probability (orig)",
    "median": "Predicted Probability (bootstrap median)",
    "lo": "CI 2.5% (bootstrap)",
    "hi": "CI 97.5% (bootstrap)",
}

CSV_FEATURE_COLS_BASELINE = {
    "age": "Age",
    "admission_NIHSS": "Admission NIHSS",
    "IVT": "IVT",
    "sex": "Male Sex",  # 1=Male, 0=Female
}

CSV_FEATURE_COLS_M3 = {
    **CSV_FEATURE_COLS_BASELINE,
    "day_NIHSS": "NIHSS 24h",
}

# ----------------------------
# Bootstrap settings
# ----------------------------
BOOTSTRAP_N = 100  # use only the first 100 bootstrap models


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)


@st.cache_resource
def load_boot_models(folder: str, n_models: int):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing bootstrap folder: {folder}")

    paths = sorted(glob.glob(os.path.join(folder, "boot_*.pkl")))
    if not paths:
        raise FileNotFoundError(f"No bootstrap models found in: {folder}")

    # take only the first N
    selected = paths[:n_models]
    if len(selected) < n_models:
        st.warning(f"Only found {len(selected)} bootstrap models in '{folder}'. Using all available.")

    return [joblib.load(p) for p in selected]


@st.cache_data
def load_plot_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def predict_proba(model, X: pd.DataFrame) -> float:
    return float(model.predict_proba(X)[0, 1])


def row_from_inputs(values: dict, cols: list) -> pd.DataFrame:
    return pd.DataFrame([[values[c] for c in cols]], columns=cols)


def fmt_pct(p: float, decimals: int = 0) -> str:
    return f"{p * 100:.{decimals}f}%"


def make_plot(x, y_main, y_med, y_lo, y_hi, xlabel, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(x, y_lo, y_hi, alpha=0.1, label="95% Model Uncertainty Interval")
    ax.plot(x, y_main, linewidth=2.7, label="Main Prediction")
    ax.plot(x, y_med, linewidth=1.3, linestyle="--", label="Bootstrap median")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def filter_plot_df(df: pd.DataFrame, inputs: dict, feats: list, x_var: str, csv_map: dict) -> pd.DataFrame:
    fixed_feats = [f for f in feats if f != x_var]
    out = df.copy()

    for f in feats:
        col = csv_map.get(f)
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for f in fixed_feats:
        col = csv_map[f]
        if col not in out.columns:
            raise ValueError(f"CSV missing required feature column: {col}")

        val = float(inputs[f])
        out = out[np.isclose(out[col].astype(float), val, atol=1e-9)]

    return out


st.subheader("Choose model")
st.write("Baseline: Before Imaging and EVT")
st.write("M3: 24 Hours After EVT")

model_choice = st.radio(
    "Model",
    options=["Baseline", "M3"],
    horizontal=True,
    label_visibility="collapsed",
)

feats = FEATURES[model_choice]

st.subheader("Enter Patient Information")
st.write("All variables must have inputs")

inputs = {}
inputs["age"] = st.number_input("Age", min_value=30, max_value=90, value=70, step=1)
inputs["admission_NIHSS"] = st.number_input("Admission NIHSS", min_value=0, max_value=50, value=10, step=1)

ivt_label = st.radio("IVT", options=["No IVT Given", "IVT Given"], horizontal=True, index=0)
inputs["IVT"] = 1 if ivt_label == "IVT Given" else 0

sex_label = st.radio("Sex", options=["Female", "Male"], horizontal=True, index=0)
inputs["sex"] = 1 if sex_label == "Male" else 0

if model_choice == "M3":
    inputs["day_NIHSS"] = st.number_input("24-hour NIHSS", min_value=0, max_value=50, value=8, step=1)

X = row_from_inputs(inputs, feats)

st.divider()
st.subheader("Probability of an unfavourable functional outcome at 90-days")

if st.button("Predict"):
    try:
        progress = st.progress(0, text="Loading models…")
        with st.spinner("Running prediction…"):
            main_model = load_model(MAIN_MODEL_PATH[model_choice])

            progress.progress(15, text=f"Loading bootstrap models (first {BOOTSTRAP_N})…")
            boot_models = load_boot_models(BOOT_DIR[model_choice], BOOTSTRAP_N)

            progress.progress(35, text="Predicting with main model…")
            p_main = predict_proba(main_model, X)

            preds = []
            n = len(boot_models)
            progress.progress(40, text=f"Running bootstraps… (0/{n})")
            for i, m in enumerate(boot_models, start=1):
                preds.append(predict_proba(m, X))
                if i % 10 == 0 or i == n:
                    pct = 40 + int(55 * (i / n))
                    progress.progress(min(pct, 95), text=f"Running bootstraps… ({i}/{n})")

            p_boot = np.asarray(preds, dtype=float)
            p_med = float(np.median(p_boot))
            p_lo = float(np.quantile(p_boot, 0.025))
            p_hi = float(np.quantile(p_boot, 0.975))

        progress.progress(100, text="Done.")
        c1, _ = st.columns(2)

        # Show as percentages
        c1.metric("Main model", fmt_pct(p_main, decimals=0))
        st.write(f"**95% Model Uncertainty Interval:** {fmt_pct(p_lo, 0)} – {fmt_pct(p_hi, 0)}")
        st.write(f"**Bootstrap Median:** {fmt_pct(p_med, 0)}")
        st.write(f"Uncertainty obtained with {len(boot_models)} model bootstraps.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.divider()
st.subheader("Probability of Unfavourable Functional Outcome at 90-days")

x_options = ["Age", "Admission NIHSS"] if model_choice == "Baseline" else ["Age", "Admission NIHSS", "24-hour NIHSS"]
x_choice = st.radio("X-axis variable", options=x_options, horizontal=True)
st.write("24-hour NIHSS only available with the M3 model.")

x_var = {"Age": "age", "Admission NIHSS": "admission_NIHSS", "24-hour NIHSS": "day_NIHSS"}[x_choice]

if st.button("Visualise Predictions"):
    try:
        df = load_plot_csv(PLOT_CSV_PATH[model_choice])

        for col in PRED_COLS.values():
            if col not in df.columns:
                raise ValueError(f"CSV missing prediction column: {col}")

        csv_map = CSV_FEATURE_COLS_M3 if model_choice == "M3" else CSV_FEATURE_COLS_BASELINE
        filt = filter_plot_df(df, inputs, feats, x_var, csv_map)

        if filt.empty:
            raise ValueError(
                "No rows found in the precomputed CSV for these fixed inputs. "
                "This usually means the grid doesn't include that exact combination."
            )

        x_col = csv_map[x_var]
        filt = filt.sort_values(by=x_col)

        x = pd.to_numeric(filt[x_col], errors="coerce").to_numpy()
        y_main = pd.to_numeric(filt[PRED_COLS["main"]], errors="coerce").to_numpy()
        y_med = pd.to_numeric(filt[PRED_COLS["median"]], errors="coerce").to_numpy()
        y_lo = pd.to_numeric(filt[PRED_COLS["lo"]], errors="coerce").to_numpy()
        y_hi = pd.to_numeric(filt[PRED_COLS["hi"]], errors="coerce").to_numpy()

        mask = np.isfinite(x) & np.isfinite(y_main) & np.isfinite(y_med) & np.isfinite(y_lo) & np.isfinite(y_hi)
        x, y_main, y_med, y_lo, y_hi = x[mask], y_main[mask], y_med[mask], y_lo[mask], y_hi[mask]

        fig = make_plot(
            x=x,
            y_main=y_main,
            y_med=y_med,
            y_lo=y_lo,
            y_hi=y_hi,
            xlabel=x_choice,
            title=f"{model_choice} Model — Probability of Unfavourable Outcome by {x_choice}",
        )
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Plot failed: {e}")
        st.info(
            "Plot uses only the precomputed CSVs. "
            "Check that your fixed inputs exist in the precomputed grid."
        )