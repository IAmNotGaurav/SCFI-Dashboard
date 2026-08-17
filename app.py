
import io
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="SCFI Monitor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# RGU VISUAL IDENTITY
# =========================================================

RGU_PURPLE = "#7B2A83"
RGU_PURPLE_DARK = "#4C1754"
RGU_PURPLE_DEEP = "#35113B"
RGU_PURPLE_LIGHT = "#F2EAF4"
RGU_CYAN = "#00A6D6"
RGU_CYAN_LIGHT = "#E6F7FC"
RGU_MAGENTA = "#A83A93"
RGU_GOLD = "#D9A928"
RGU_TEXT = "#24212A"
RGU_MUTED = "#6D6873"
RGU_BORDER = "#DDD5E0"
RGU_SURFACE = "#FFFFFF"
RGU_BACKGROUND = "#F7F5F8"

RGU_CHART_COLORS = [
    RGU_PURPLE,
    RGU_CYAN,
    RGU_MAGENTA,
    RGU_GOLD,
    "#5E8C61",
    "#8B6BA3",
]

rgu_plotly_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=RGU_SURFACE,
        plot_bgcolor=RGU_SURFACE,
        font=dict(
            family="Arial, Helvetica, sans-serif",
            color=RGU_TEXT,
            size=13
        ),
        title=dict(
            font=dict(
                family="Arial, Helvetica, sans-serif",
                color=RGU_PURPLE_DARK,
                size=19
            ),
            x=0.02
        ),
        colorway=RGU_CHART_COLORS,
        xaxis=dict(
            gridcolor="#EEE9F0",
            linecolor=RGU_BORDER,
            zerolinecolor=RGU_BORDER,
            title_font=dict(color=RGU_PURPLE_DARK)
        ),
        yaxis=dict(
            gridcolor="#EEE9F0",
            linecolor=RGU_BORDER,
            zerolinecolor=RGU_BORDER,
            title_font=dict(color=RGU_PURPLE_DARK)
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(color=RGU_TEXT)
        ),
        hoverlabel=dict(
            bgcolor=RGU_PURPLE_DARK,
            font_color="#FFFFFF"
        )
    )
)

pio.templates["rgu"] = rgu_plotly_template
pio.templates.default = "rgu"
px.defaults.color_discrete_sequence = RGU_CHART_COLORS

st.markdown(
    f"""
    <style>
      :root {{
          --rgu-purple: {RGU_PURPLE};
          --rgu-purple-dark: {RGU_PURPLE_DARK};
          --rgu-purple-deep: {RGU_PURPLE_DEEP};
          --rgu-purple-light: {RGU_PURPLE_LIGHT};
          --rgu-cyan: {RGU_CYAN};
          --rgu-cyan-light: {RGU_CYAN_LIGHT};
          --rgu-text: {RGU_TEXT};
          --rgu-muted: {RGU_MUTED};
          --rgu-border: {RGU_BORDER};
          --rgu-bg: {RGU_BACKGROUND};
      }}

      .stApp {{
          background: var(--rgu-bg);
          color: var(--rgu-text);
      }}

      .block-container {{
          padding-top: 1.25rem;
          padding-bottom: 3rem;
          max-width: 1500px;
      }}

      h1, h2, h3, h4 {{
          color: var(--rgu-purple-dark) !important;
          letter-spacing: -0.01em;
      }}

      p, li, label {{
          color: var(--rgu-text);
      }}

      [data-testid="stSidebar"] {{
          background:
            linear-gradient(
              180deg,
              var(--rgu-purple-deep) 0%,
              var(--rgu-purple-dark) 58%,
              var(--rgu-purple) 100%
            );
          border-right: none;
      }}

      [data-testid="stSidebar"] * {{
          color: #FFFFFF;
      }}

      [data-testid="stSidebar"] hr {{
          border-color: rgba(255,255,255,.18);
      }}

      [data-testid="stSidebar"] [data-testid="stFileUploader"] {{
          background: rgba(255,255,255,.08);
          border: 1px solid rgba(255,255,255,.18);
          border-radius: 12px;
          padding: 8px;
      }}

      .hero {{
          position: relative;
          overflow: hidden;
          padding: 30px 34px;
          border-radius: 0 0 18px 18px;
          background:
            linear-gradient(
              105deg,
              var(--rgu-purple-deep) 0%,
              var(--rgu-purple-dark) 54%,
              var(--rgu-purple) 100%
            );
          border: none;
          border-bottom: 5px solid var(--rgu-cyan);
          margin: -1.25rem 0 24px 0;
          box-shadow: 0 8px 24px rgba(69, 24, 78, .12);
      }}

      .hero:after {{
          content: "";
          position: absolute;
          right: -80px;
          top: -115px;
          width: 300px;
          height: 300px;
          border: 38px solid rgba(255,255,255,.07);
          border-radius: 50%;
      }}

      .hero h1 {{
          position: relative;
          z-index: 2;
          margin: 2px 0 0 0;
          font-size: 2.25rem;
          color: #FFFFFF !important;
      }}

      .hero p {{
          position: relative;
          z-index: 2;
          margin: 10px 0 0 0;
          color: rgba(255,255,255,.84);
          font-size: 1rem;
          max-width: 1050px;
      }}

      .section-label {{
          position: relative;
          z-index: 2;
          display: inline-block;
          font-size: .74rem;
          text-transform: uppercase;
          letter-spacing: .11em;
          color: #FFFFFF;
          background: rgba(0,166,214,.22);
          border: 1px solid rgba(124,224,250,.5);
          padding: 6px 9px;
          border-radius: 999px;
          margin-bottom: 8px;
      }}

      [data-testid="stMetric"] {{
          background: #FFFFFF;
          border: 1px solid var(--rgu-border);
          border-top: 4px solid var(--rgu-purple);
          padding: 17px 18px 15px 18px;
          border-radius: 11px;
          box-shadow: 0 5px 16px rgba(55, 28, 59, .055);
      }}

      [data-testid="stMetricLabel"] {{
          color: var(--rgu-muted) !important;
          font-size: .86rem;
          font-weight: 600;
      }}

      [data-testid="stMetricValue"] {{
          color: var(--rgu-purple-dark) !important;
      }}

      .stTabs [data-baseweb="tab-list"] {{
          gap: 4px;
          border-bottom: 1px solid var(--rgu-border);
      }}

      .stTabs [data-baseweb="tab"] {{
          height: 48px;
          padding: 0 14px;
          color: var(--rgu-muted);
          font-weight: 600;
      }}

      .stTabs [aria-selected="true"] {{
          color: var(--rgu-purple-dark) !important;
          background: var(--rgu-purple-light);
          border-radius: 8px 8px 0 0;
      }}

      .stTabs [data-baseweb="tab-highlight"] {{
          background-color: var(--rgu-purple) !important;
          height: 3px;
      }}

      .stButton > button,
      .stDownloadButton > button {{
          border-radius: 8px;
          border: 1px solid var(--rgu-purple);
          font-weight: 700;
      }}

      .stButton > button[kind="primary"] {{
          background: var(--rgu-purple) !important;
          border-color: var(--rgu-purple) !important;
          color: #FFFFFF !important;
      }}

      .stDownloadButton > button {{
          background: #FFFFFF;
          color: var(--rgu-purple-dark);
      }}

      .stDownloadButton > button:hover {{
          background: var(--rgu-purple-light);
          border-color: var(--rgu-purple-dark);
      }}

      [data-testid="stDataFrame"] {{
          background: #FFFFFF;
          border: 1px solid var(--rgu-border);
          border-radius: 10px;
          overflow: hidden;
      }}

      [data-testid="stExpander"] {{
          background: #FFFFFF;
          border: 1px solid var(--rgu-border);
          border-radius: 10px;
      }}

      .insight-box {{
          padding: 15px 17px;
          border-radius: 9px;
          background: #FFFFFF;
          border: 1px solid var(--rgu-border);
          border-left: 5px solid var(--rgu-purple);
          color: var(--rgu-text);
          margin-bottom: 10px;
          box-shadow: 0 3px 10px rgba(55,28,59,.04);
      }}

      .small-note {{
          color: var(--rgu-muted);
          font-size: .88rem;
      }}

      .risk-moderate, .risk-high, .risk-severe {{
          padding: 12px 16px;
          border-radius: 9px;
          font-weight: 800;
          text-align: left;
          margin: 8px 0 17px 0;
          background: #FFFFFF;
      }}

      .risk-moderate {{
          border: 1px solid #9DC9A7;
          border-left: 6px solid #3B8D53;
          color: #246B39;
      }}

      .risk-high {{
          border: 1px solid #E2C66D;
          border-left: 6px solid #C79500;
          color: #7B5A00;
      }}

      .risk-severe {{
          border: 1px solid #E5A39C;
          border-left: 6px solid #C4483E;
          color: #8E2E27;
      }}

      div[data-baseweb="select"] > div,
      div[data-baseweb="input"] > div {{
          background: #FFFFFF;
          border-color: var(--rgu-border);
      }}

      .rgu-footer {{
          margin-top: 34px;
          padding: 16px 4px 4px 4px;
          border-top: 3px solid var(--rgu-purple);
          color: var(--rgu-muted);
          font-size: .82rem;
      }}

      .rgu-footer strong {{
          color: var(--rgu-purple-dark);
      }}
    </style>
    """,
    unsafe_allow_html=True
)

VARIABLES = ["GSCPI", "Inflation_Rate", "GPR", "Oil_Return"]
FEATURE_COLUMNS = VARIABLES + ["SCFI"]
LAGS = [1, 2, 3, 4, 5, 6]

EVENTS = {
    "COVID-19 pandemic": "2020-03",
    "Suez Canal blockage": "2021-03",
    "Russia–Ukraine war": "2022-02",
}


# =========================================================
# HELPERS
# =========================================================

def classify_fragility(value):
    if value < 0.5:
        return "Moderate"
    elif value < 1.5:
        return "High"
    return "Severe"


def risk_html(label, text):
    css = {
        "Moderate": "risk-moderate",
        "High": "risk-high",
        "Severe": "risk-severe"
    }.get(label, "risk-high")
    return f'<div class="{css}">{text}</div>'


def read_uploaded_file(uploaded_file, sheet_name=None):
    uploaded_file.seek(0)
    lower = uploaded_file.name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if lower.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)
    raise ValueError("Upload CSV, XLS or XLSX files only.")


def excel_sheet_names(uploaded_file):
    lower = uploaded_file.name.lower()
    if not lower.endswith((".xls", ".xlsx")):
        return []
    uploaded_file.seek(0)
    return pd.ExcelFile(uploaded_file).sheet_names


def clean_two_column_data(raw, date_col, value_col, output_name, skip_rows=0):
    df = raw.iloc[int(skip_rows):][[date_col, value_col]].copy()
    df.columns = ["Date", output_name]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[output_name] = pd.to_numeric(df[output_name], errors="coerce")

    df = (
        df.dropna(subset=["Date", output_name])
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    df["Date"] = df["Date"].dt.to_period("M")

    # Normalise duplicates at monthly level
    return df.groupby("Date", as_index=False)[output_name].mean()


def clean_oil_data(raw, date_col, value_col, oil_mode, skip_rows=0):
    oil = raw.iloc[int(skip_rows):][[date_col, value_col]].copy()
    oil.columns = ["Date", "Oil_Value"]
    oil["Date"] = pd.to_datetime(oil["Date"], errors="coerce")
    oil["Oil_Value"] = pd.to_numeric(oil["Oil_Value"], errors="coerce")

    oil = (
        oil.dropna(subset=["Date", "Oil_Value"])
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if oil_mode == "Oil price — calculate monthly return":
        monthly = (
            oil.set_index("Date")
            .resample("ME")
            .mean()
            .reset_index()
        )
        monthly["Oil_Return"] = monthly["Oil_Value"].pct_change() * 100
        result = monthly[["Date", "Oil_Return"]].dropna().copy()
    else:
        result = oil.rename(columns={"Oil_Value": "Oil_Return"})
        result = (
            result.set_index("Date")
            .resample("ME")
            .mean()
            .reset_index()[["Date", "Oil_Return"]]
        )

    result["Date"] = result["Date"].dt.to_period("M")
    return result


def create_lagged_dataset(master_scaled):
    forecast_data = master_scaled.copy()
    lagged_features = []

    for col in FEATURE_COLUMNS:
        for lag in LAGS:
            name = f"{col}_lag{lag}"
            forecast_data[name] = forecast_data[col].shift(lag)
            lagged_features.append(name)

    next_month_scfi = master_scaled.set_index("Date")["SCFI"].shift(-1)
    forecast_data["Target_SCFI"] = forecast_data["Date"].map(next_month_scfi)
    forecast_data["Year"] = forecast_data["Date"].dt.year

    regression_data = (
        forecast_data
        .dropna(subset=lagged_features + ["Target_SCFI"])
        .reset_index(drop=True)
    )

    return forecast_data, regression_data, lagged_features


def build_regressor(model_name):
    if model_name == "Linear Regression":
        return LinearRegression(), True

    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ), False

    if model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError(
                "XGBoost is not installed. Add xgboost to requirements.txt."
            )

        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1
        ), False

    raise ValueError("Unknown model.")


def walk_forward_regression(regression_data, lagged_features, model_name):
    years = sorted(
        regression_data.loc[
            regression_data["Year"] >= 2019,
            "Year"
        ].unique()
    )

    actual, predicted, origins = [], [], []

    for year in years:
        train = regression_data[regression_data["Year"] < year].copy()
        test = regression_data[regression_data["Year"] == year].copy()

        if train.empty or test.empty:
            continue

        X_train = train[lagged_features]
        X_test = test[lagged_features]
        y_train = train["Target_SCFI"]
        y_test = test["Target_SCFI"]

        model, requires_scaling = build_regressor(model_name)

        if requires_scaling:
            scaler = StandardScaler()
            X_train_model = scaler.fit_transform(X_train)
            X_test_model = scaler.transform(X_test)
        else:
            X_train_model = X_train
            X_test_model = X_test

        model.fit(X_train_model, y_train)
        preds = model.predict(X_test_model)

        actual.extend(y_test.tolist())
        predicted.extend(np.asarray(preds).tolist())
        origins.extend(test["Date"].tolist())

    if not actual:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R²": np.nan,
            "Residual Std": np.nan,
            "Predictions": pd.DataFrame()
        }

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    residuals = np.array(actual) - np.array(predicted)

    pred_df = pd.DataFrame({
        "Forecast_Origin": origins,
        "Target_Date": [d + 1 for d in origins],
        "Actual_SCFI": actual,
        "Predicted_SCFI": predicted,
        "Residual": residuals
    })

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Residual Std": residuals.std(ddof=1),
        "Predictions": pred_df
    }


def final_model_forecast(
    regression_data,
    forecast_data,
    lagged_features,
    model_name
):
    X_train = regression_data[lagged_features]
    y_train = regression_data["Target_SCFI"]

    model, requires_scaling = build_regressor(model_name)

    scaler = None
    if requires_scaling:
        scaler = StandardScaler()
        X_model = scaler.fit_transform(X_train)
    else:
        X_model = X_train

    model.fit(X_model, y_train)

    latest = (
        forecast_data
        .dropna(subset=lagged_features)
        .iloc[[-1]]
        .copy()
    )

    X_latest = latest[lagged_features]

    if requires_scaling:
        X_latest_model = scaler.transform(X_latest)
    else:
        X_latest_model = X_latest

    forecast = float(model.predict(X_latest_model)[0])

    return {
        "Model": model,
        "Scaler": scaler,
        "Forecast": forecast,
        "Origin": latest.iloc[0]["Date"],
        "Target": latest.iloc[0]["Date"] + 1
    }


def analyse_uploaded_data(gscpi, inflation, gpr, oil):
    quality = []
    for name, df, col in [
        ("GSCPI", gscpi, "GSCPI"),
        ("Inflation", inflation, "Inflation_Rate"),
        ("GPR", gpr, "GPR"),
        ("Oil Return", oil, "Oil_Return"),
    ]:
        quality.append({
            "Dataset": name,
            "Rows": len(df),
            "Start": str(df["Date"].min()),
            "End": str(df["Date"].max()),
            "Missing": int(df[col].isna().sum())
        })

    master = (
        gscpi.merge(inflation, on="Date", how="inner")
        .merge(gpr, on="Date", how="inner")
        .merge(oil, on="Date", how="inner")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(master) < 36:
        raise ValueError(
            "The common merged data contains too few monthly observations."
        )

    # Standardise four index components
    index_scaler = StandardScaler()
    scaled = index_scaler.fit_transform(master[VARIABLES])

    master_scaled = master[["Date"]].copy()
    master_scaled[VARIABLES] = scaled

    # PCA
    pca = PCA()
    pca.fit(master_scaled[VARIABLES])

    loading = pca.components_[0]
    abs_loading = np.abs(loading)
    weights = abs_loading / abs_loading.sum()

    weight_map = dict(zip(VARIABLES, weights))

    master_scaled["SCFI"] = sum(
        weight_map[col] * master_scaled[col]
        for col in VARIABLES
    )
    master_scaled["Fragility_Class"] = (
        master_scaled["SCFI"].apply(classify_fragility)
    )

    correlation = master_scaled[["SCFI", "GSCPI"]].corr().iloc[0, 1]
    val_mae = mean_absolute_error(master_scaled["GSCPI"], master_scaled["SCFI"])
    val_rmse = np.sqrt(
        mean_squared_error(master_scaled["GSCPI"], master_scaled["SCFI"])
    )

    forecast_data, regression_data, lagged_features = (
        create_lagged_dataset(master_scaled)
    )

    # Walk-forward regression comparison
    available_models = ["Linear Regression", "Random Forest"]
    if XGBOOST_AVAILABLE:
        available_models.append("XGBoost")

    model_rows = []
    wf_results = {}

    for name in available_models:
        wf = walk_forward_regression(
            regression_data,
            lagged_features,
            name
        )
        wf_results[name] = wf
        model_rows.append({
            "Model": name,
            "MAE": wf["MAE"],
            "RMSE": wf["RMSE"],
            "R²": wf["R²"]
        })

    model_comparison = (
        pd.DataFrame(model_rows)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    best_model = model_comparison.iloc[0]["Model"]

    # Model forecasts for selector
    final_forecasts = {}
    for name in available_models:
        f = final_model_forecast(
            regression_data,
            forecast_data,
            lagged_features,
            name
        )
        residual_std = wf_results[name]["Residual Std"]
        f["Lower"] = (
            f["Forecast"] - 1.96 * residual_std
            if np.isfinite(residual_std) else np.nan
        )
        f["Upper"] = (
            f["Forecast"] + 1.96 * residual_std
            if np.isfinite(residual_std) else np.nan
        )
        f["Fragility"] = classify_fragility(f["Forecast"])
        final_forecasts[name] = f

    pca_table = pd.DataFrame({
        "Indicator": VARIABLES,
        "PC1 Loading": loading,
        "Absolute Loading": abs_loading,
        "Weight": weights,
        "Weight (%)": weights * 100
    })

    # Latest raw observations
    latest_raw = master.iloc[-1].copy()

    return {
        "quality": pd.DataFrame(quality),
        "master": master,
        "master_scaled": master_scaled,
        "index_scaler": index_scaler,
        "pca": pca,
        "weights": weight_map,
        "pca_table": pca_table,
        "correlation": correlation,
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "forecast_data": forecast_data,
        "regression_data": regression_data,
        "lagged_features": lagged_features,
        "model_comparison": model_comparison,
        "wf_results": wf_results,
        "final_forecasts": final_forecasts,
        "best_model": best_model,
        "latest_raw": latest_raw
    }


def change_indicator(value_now, value_prev, inverse=False):
    if value_prev is None or pd.isna(value_prev):
        return None

    difference = value_now - value_prev

    if inverse:
        difference *= -1

    return difference


def generate_insights(results, selected_model):
    ms = results["master_scaled"]
    current = float(ms.iloc[-1]["SCFI"])
    prev = float(ms.iloc[-2]["SCFI"])
    forecast = results["final_forecasts"][selected_model]["Forecast"]

    movement = current - prev
    forecast_change = forecast - current

    lines = []

    if movement > 0.10:
        lines.append(
            "The current SCFI has increased materially from the previous month, "
            "indicating rising near-term fragility."
        )
    elif movement < -0.10:
        lines.append(
            "The current SCFI has declined from the previous month, indicating "
            "an easing of observed fragility."
        )
    else:
        lines.append(
            "The current SCFI is broadly stable relative to the previous month."
        )

    if forecast_change < -0.10:
        lines.append(
            f"The {selected_model} forecast indicates an improvement next month, "
            f"with SCFI expected to fall by {abs(forecast_change):.2f} points."
        )
    elif forecast_change > 0.10:
        lines.append(
            f"The {selected_model} forecast indicates higher fragility next month, "
            f"with SCFI expected to rise by {forecast_change:.2f} points."
        )
    else:
        lines.append(
            f"The {selected_model} forecast suggests limited month-to-month change."
        )

    biggest = (
        results["pca_table"]
        .sort_values("Weight (%)", ascending=False)
        .iloc[0]
    )
    lines.append(
        f"{biggest['Indicator']} carries the largest PCA-derived weight "
        f"({biggest['Weight (%)']:.1f}%) in the uploaded dataset."
    )

    return lines


def create_excel_report(results, selected_model, scenario_table=None):
    output = io.BytesIO()

    master = results["master"].copy()
    master["Date"] = master["Date"].astype(str)

    scaled = results["master_scaled"].copy()
    scaled["Date"] = scaled["Date"].astype(str)

    validation = pd.DataFrame({
        "Metric": [
            "Pearson correlation SCFI vs GSCPI",
            "SCFI vs GSCPI MAE",
            "SCFI vs GSCPI RMSE",
            "PC1 explained variance"
        ],
        "Value": [
            results["correlation"],
            results["validation_mae"],
            results["validation_rmse"],
            results["pca"].explained_variance_ratio_[0]
        ]
    })

    f = results["final_forecasts"][selected_model]
    forecast = pd.DataFrame({
        "Selected Model": [selected_model],
        "Forecast Origin": [str(f["Origin"])],
        "Forecast Month": [str(f["Target"])],
        "Forecasted SCFI": [f["Forecast"]],
        "Lower 95% Approx": [f["Lower"]],
        "Upper 95% Approx": [f["Upper"]],
        "Fragility": [f["Fragility"]]
    })

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results["quality"].to_excel(writer, sheet_name="Data Quality", index=False)
        master.to_excel(writer, sheet_name="Merged Data", index=False)
        scaled.to_excel(writer, sheet_name="SCFI History", index=False)
        results["pca_table"].to_excel(writer, sheet_name="PCA Weights", index=False)
        validation.to_excel(writer, sheet_name="Validation", index=False)
        results["model_comparison"].to_excel(writer, sheet_name="Regression Models", index=False
        )
        forecast.to_excel(writer, sheet_name="Selected Forecast", index=False)

        for model, wf in results["wf_results"].items():
            if not wf["Predictions"].empty:
                tmp = wf["Predictions"].copy()
                tmp["Forecast_Origin"] = tmp["Forecast_Origin"].astype(str)
                tmp["Target_Date"] = tmp["Target_Date"].astype(str)
                tmp.to_excel(
                    writer,
                    sheet_name=f"{model[:20]} WF",
                    index=False
                )

        if scenario_table is not None:
            scenario_table.to_excel(writer, sheet_name="Scenario Simulator",
                index=False
            )

    output.seek(0)
    return output.getvalue()


def create_pdf_report(results, selected_model, insights):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="DashboardTitle",
            parent=styles["Heading1"],
            fontSize=19,
            leading=23,
            textColor=colors.HexColor("#4C1754"),
            alignment=TA_CENTER,
            spaceAfter=12
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#4C1754"),
            spaceBefore=10,
            spaceAfter=6
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCompact",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13
        )
    )

    story = [
        Paragraph(
            "Supply Chain Fragility Index – Analytical Report",
            styles["DashboardTitle"]
        ),
        Paragraph(
            f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')}",
            styles["BodyCompact"]
        ),
        Spacer(1, 8)
    ]

    current = results["master_scaled"].iloc[-1]
    f = results["final_forecasts"][selected_model]

    summary_data = [
        ["Metric", "Value"],
        ["Latest common month", str(current["Date"])],
        ["Current SCFI", f"{current['SCFI']:.4f}"],
        ["Current fragility", current["Fragility_Class"]],
        ["Selected model", selected_model],
        ["Forecast month", str(f["Target"])],
        ["Forecast SCFI", f"{f['Forecast']:.4f}"],
        ["Forecast fragility", f["Fragility"]],
    ]

    t = Table(summary_data, colWidths=[75 * mm, 85 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C1754")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#C8D0DB")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#F4F7FA")
        ])
    ]))
    story.extend([
        Paragraph("Executive Summary", styles["ReportHeading"]),
        t,
        Spacer(1, 8),
        Paragraph("Automated interpretation", styles["ReportHeading"])
    ])

    for insight in insights:
        story.append(
            Paragraph("• " + insight, styles["BodyCompact"])
        )

    story.append(
        Paragraph("PCA-derived weights", styles["ReportHeading"])
    )

    weight_data = [["Indicator", "Weight (%)"]]
    for _, row in results["pca_table"].iterrows():
        weight_data.append(
            [row["Indicator"], f"{row['Weight (%)']:.2f}"]
        )

    wt = Table(weight_data, colWidths=[85 * mm, 55 * mm])
    wt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C1754")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#C8D0DB")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(wt)

    story.append(
        Paragraph("Regression model comparison", styles["ReportHeading"])
    )

    model_data = [["Model", "MAE", "RMSE", "R²"]]
    for _, row in results["model_comparison"].iterrows():
        model_data.append([
            row["Model"],
            f"{row['MAE']:.4f}",
            f"{row['RMSE']:.4f}",
            f"{row['R²']:.4f}",
        ])

    mt = Table(model_data, colWidths=[60 * mm, 30 * mm, 30 * mm, 30 * mm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C1754")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#C8D0DB")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ]))
    story.append(mt)

    story.extend([
        Spacer(1, 10),
        Paragraph(
            "<b>Interpretation note:</b> Forecasts are analytical decision-support "
            "estimates rather than certainty. Results depend on the quality and "
            "timeliness of the uploaded datasets.",
            styles["BodyCompact"]
        )
    ])

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# SESSION STATE
# =========================================================

if "results" not in st.session_state:
    st.session_state.results = None


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
      <div class="section-label">Robert Gordon University · MSc Business Analytics</div>
      <h1>Supply Chain Fragility Monitor</h1>
      <p>
        A decision-support application for constructing, validating and forecasting
        the Supply Chain Fragility Index (SCFI) using operational, macroeconomic,
        geopolitical and energy-market signals.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR: UPLOADS
# =========================================================

with st.sidebar:
    st.header("Data workspace")
    st.caption("Upload the four datasets used by the SCFI framework.")

    gscpi_file = st.file_uploader(
        "GSCPI",
        type=["csv", "xls", "xlsx"],
        key="gscpi"
    )
    inflation_file = st.file_uploader(
        "Inflation",
        type=["csv", "xls", "xlsx"],
        key="inflation"
    )
    gpr_file = st.file_uploader(
        "Geopolitical Risk (GPR)",
        type=["csv", "xls", "xlsx"],
        key="gpr"
    )
    oil_file = st.file_uploader(
        "WTI Oil",
        type=["csv", "xls", "xlsx"],
        key="oil"
    )

    st.divider()

    if st.button("Reset application", width="stretch"):
        st.session_state.results = None
        st.rerun()


uploads_ready = all(
    x is not None
    for x in [gscpi_file, inflation_file, gpr_file, oil_file]
)

if not uploads_ready:
    intro1, intro2, intro3 = st.columns(3)
    with intro1:
        st.info(
            "**1 · Upload**\n\nProvide GSCPI, inflation, GPR and WTI oil data."
        )
    with intro2:
        st.info(
            "**2 · Analyse**\n\nPCA weights, SCFI, validation and forecasting are calculated automatically."
        )
    with intro3:
        st.info(
            "**3 · Decide**\n\nExplore risk, simulate scenarios and download executive reports."
        )

    st.markdown("### What the platform delivers")
    st.markdown(
        """
        - Current **Moderate / High / Severe** fragility status  
        - Interactive SCFI history with major disruption markers  
        - PCA contribution analysis across the four indicators  
        - Walk-forward comparison of **Linear Regression, Random Forest and XGBoost**  
        - One-month-ahead forecast with approximate uncertainty interval  
        - What-if scenario simulation  
        - Excel, PDF and HTML-ready analytical outputs
        """
    )
    st.stop()


# =========================================================
# UPLOAD CONFIGURATION
# =========================================================

with st.expander("Configure uploaded datasets", expanded=True):
    tabs = st.tabs(["GSCPI", "Inflation", "GPR", "Oil"])

    def configure(
        uploaded,
        prefix,
        suggested_date,
        suggested_value,
        default_skip=0,
        preferred_sheet=None
    ):
        sheets = excel_sheet_names(uploaded)
        sheet = None

        if sheets:
            default_idx = (
                sheets.index(preferred_sheet)
                if preferred_sheet in sheets
                else 0
            )
            sheet = st.selectbox(
                "Worksheet",
                sheets,
                index=default_idx,
                key=f"{prefix}_sheet"
            )

        raw = read_uploaded_file(uploaded, sheet)
        cols = list(raw.columns)

        date_index = cols.index(suggested_date) if suggested_date in cols else 0
        value_index = (
            cols.index(suggested_value)
            if suggested_value in cols
            else min(1, len(cols) - 1)
        )

        date_col = st.selectbox(
            "Date column",
            cols,
            index=date_index,
            key=f"{prefix}_date"
        )
        value_col = st.selectbox(
            "Value column",
            cols,
            index=value_index,
            key=f"{prefix}_value"
        )
        skip = st.number_input(
            "Rows to skip before data starts",
            min_value=0,
            max_value=max(0, len(raw) - 1),
            value=min(default_skip, max(0, len(raw) - 1)),
            step=1,
            key=f"{prefix}_skip"
        )

        st.caption(
            f"Detected {len(raw):,} rows × {len(cols)} columns."
        )
        st.dataframe(raw.head(8), width="stretch")

        return raw, date_col, value_col, skip

    with tabs[0]:
        g_raw, g_date, g_value, g_skip = configure(
            gscpi_file,
            "gscpi_cfg",
            "Date",
            "GSCPI",
            default_skip=4,
            preferred_sheet="GSCPI Monthly Data"
        )

    with tabs[1]:
        i_raw, i_date, i_value, i_skip = configure(
            inflation_file,
            "inflation_cfg",
            "TIME_PERIOD",
            "OBS_VALUE",
            default_skip=0
        )

    with tabs[2]:
        gp_raw, gp_date, gp_value, gp_skip = configure(
            gpr_file,
            "gpr_cfg",
            "month",
            "GPR",
            default_skip=0,
            preferred_sheet="Sheet1"
        )

    with tabs[3]:
        o_raw, o_date, o_value, o_skip = configure(
            oil_file,
            "oil_cfg",
            "observation_date",
            "DCOILWTICO",
            default_skip=0
        )
        oil_mode = st.radio(
            "Uploaded oil value",
            [
                "Oil price — calculate monthly return",
                "Monthly oil return already calculated"
            ],
            key="oil_mode"
        )


if st.button(
    "Validate data and build SCFI",
    type="primary",
    width="stretch"
):
    try:
        with st.spinner(
            "Validating files, constructing SCFI and evaluating forecasting models..."
        ):
            gscpi = clean_two_column_data(
                g_raw, g_date, g_value, "GSCPI", g_skip
            )
            inflation = clean_two_column_data(
                i_raw, i_date, i_value, "Inflation_Rate", i_skip
            )
            gpr = clean_two_column_data(
                gp_raw, gp_date, gp_value, "GPR", gp_skip
            )
            oil = clean_oil_data(
                o_raw, o_date, o_value, oil_mode, o_skip
            )

            st.session_state.results = analyse_uploaded_data(
                gscpi, inflation, gpr, oil
            )

        st.success("Analysis completed successfully.")

    except Exception as exc:
        st.session_state.results = None
        st.error("The uploaded data could not be analysed.")
        st.exception(exc)


results = st.session_state.results

if results is None:
    st.stop()


# =========================================================
# MODEL CONTROL
# =========================================================

available_models = results["model_comparison"]["Model"].tolist()
default_model = (
    "Linear Regression"
    if "Linear Regression" in available_models
    else available_models[0]
)

selected_model = st.sidebar.selectbox(
    "Forecast model",
    available_models,
    index=available_models.index(default_model),
    help="Linear Regression was the best model in the final dissertation notebook."
)

selected_forecast = results["final_forecasts"][selected_model]
current_row = results["master_scaled"].iloc[-1]
previous_row = results["master_scaled"].iloc[-2]
latest_raw = results["latest_raw"]


# =========================================================
# MAIN NAVIGATION
# =========================================================

pages = st.tabs([
    "Executive Dashboard",
    "Data Quality",
    "SCFI Analysis",
    "Event Explorer",
    "Forecasting",
    "Scenario Simulator",
    "Reports & Method"
])


# =========================================================
# EXECUTIVE DASHBOARD
# =========================================================

with pages[0]:
    st.markdown("## Executive Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    current_scfi = float(current_row["SCFI"])
    previous_scfi = float(previous_row["SCFI"])

    c1.metric(
        "Current SCFI",
        f"{current_scfi:.3f}",
        delta=f"{current_scfi - previous_scfi:+.3f} vs prior month"
    )
    c2.metric(
        "Current fragility",
        current_row["Fragility_Class"]
    )
    c3.metric(
        f"Forecast SCFI · {selected_forecast['Target']}",
        f"{selected_forecast['Forecast']:.3f}",
        delta=f"{selected_forecast['Forecast'] - current_scfi:+.3f}"
    )
    c4.metric(
        "Forecast fragility",
        selected_forecast["Fragility"]
    )

    st.markdown(
        risk_html(
            selected_forecast["Fragility"],
            f"Next-month outlook: {selected_forecast['Fragility'].upper()} "
            f"fragility using {selected_model}"
        ),
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # FRAGILITY CLASS EXPLANATION
    # Moved from the methodology section so a first-time user
    # can understand the warning classes directly on the
    # Executive Dashboard.
    # ---------------------------------------------------------
    st.markdown("### Understanding the fragility classes")
    st.markdown(
        """
        SCFI is a **continuous fragility measure**. The three classes below are
        used as an operational warning layer so the index is easier to interpret.
        A higher SCFI value indicates a broader combination of operational,
        macroeconomic, geopolitical and energy-market pressure.
        """
    )

    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        st.markdown(
            """
            <div style="
                border: 1px solid rgba(46,139,109,0.35);
                border-top: 5px solid #2E8B6D;
                border-radius: 12px;
                padding: 16px 18px;
                min-height: 178px;
                background: rgba(46,139,109,0.06);
            ">
                <div style="font-size:0.82rem;font-weight:800;letter-spacing:.08em;color:#2E8B6D;">
                    MODERATE
                </div>
                <div style="font-size:1.35rem;font-weight:800;margin:5px 0 9px 0;">
                    SCFI &lt; 0.5
                </div>
                <div style="line-height:1.45;">
                    Fragility is closer to normal historical variability.
                    Individual pressures may still exist, but the combined
                    multidimensional signal is not strongly elevated.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with fc2:
        st.markdown(
            """
            <div style="
                border: 1px solid rgba(204,151,0,0.38);
                border-top: 5px solid #CC9700;
                border-radius: 12px;
                padding: 16px 18px;
                min-height: 178px;
                background: rgba(204,151,0,0.07);
            ">
                <div style="font-size:0.82rem;font-weight:800;letter-spacing:.08em;color:#A87800;">
                    HIGH
                </div>
                <div style="font-size:1.35rem;font-weight:800;margin:5px 0 9px 0;">
                    0.5 ≤ SCFI &lt; 1.5
                </div>
                <div style="line-height:1.45;">
                    Fragility is elevated. More than one pressure dimension may
                    be contributing and the environment should be monitored
                    more closely for further deterioration.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with fc3:
        st.markdown(
            """
            <div style="
                border: 1px solid rgba(184,74,67,0.38);
                border-top: 5px solid #B84A43;
                border-radius: 12px;
                padding: 16px 18px;
                min-height: 178px;
                background: rgba(184,74,67,0.07);
            ">
                <div style="font-size:0.82rem;font-weight:800;letter-spacing:.08em;color:#B84A43;">
                    SEVERE
                </div>
                <div style="font-size:1.35rem;font-weight:800;margin:5px 0 9px 0;">
                    SCFI ≥ 1.5
                </div>
                <div style="line-height:1.45;">
                    Fragility is unusually broad and intense. Operational,
                    economic, geopolitical and/or energy pressures are combining
                    strongly enough to indicate a severe warning condition.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    current_class = current_row["Fragility_Class"]
    forecast_class = selected_forecast["Fragility"]

    st.markdown(
        f"""
        <div class="insight-box">
            <strong>How to read the dashboard now:</strong>
            the latest observed SCFI is <strong>{current_scfi:.3f}</strong>,
            which is classified as <strong>{current_class}</strong>.
            The next-month forecast is <strong>{selected_forecast['Forecast']:.3f}</strong>,
            which is classified as <strong>{forecast_class}</strong>.
            The class gives a quick warning level, while the continuous SCFI value
            shows how close the result is to the next threshold.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Latest indicator conditions")
    k1, k2, k3, k4 = st.columns(4)

    raw = results["master"]
    prev_raw = raw.iloc[-2]

    k1.metric(
        "GSCPI",
        f"{latest_raw['GSCPI']:.3f}",
        delta=f"{latest_raw['GSCPI'] - prev_raw['GSCPI']:+.3f}"
    )
    k2.metric(
        "Inflation",
        f"{latest_raw['Inflation_Rate']:.2f}%",
        delta=f"{latest_raw['Inflation_Rate'] - prev_raw['Inflation_Rate']:+.2f} pp"
    )
    k3.metric(
        "Geopolitical Risk",
        f"{latest_raw['GPR']:.1f}",
        delta=f"{latest_raw['GPR'] - prev_raw['GPR']:+.1f}"
    )
    k4.metric(
        "Oil monthly return",
        f"{latest_raw['Oil_Return']:.2f}%",
        delta=f"{latest_raw['Oil_Return'] - prev_raw['Oil_Return']:+.2f} pp"
    )

    left, right = st.columns([2.1, 1])

    with left:
        hist = results["master_scaled"].tail(120)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist["Date"].astype(str),
                y=hist["SCFI"],
                mode="lines",
                name="SCFI",
                line=dict(width=3)
            )
        )

        for event, date in EVENTS.items():
            if date in hist["Date"].astype(str).values:
                fig.add_shape(
                    type="line",
                    x0=date,
                    x1=date,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(dash="dash", width=1)
                )
                fig.add_annotation(
                    x=date,
                    y=1,
                    xref="x",
                    yref="paper",
                    text=event,
                    showarrow=False,
                    yshift=12,
                    textangle=-35
                )

        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=.5,
            y1=.5,
            xref="paper",
            yref="y",
            line=dict(dash="dot", width=1)
        )
        fig.add_annotation(
            x=1,
            y=.5,
            xref="paper",
            yref="y",
            text="High threshold",
            showarrow=False,
            xanchor="right",
            yshift=10
        )

        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=1.5,
            y1=1.5,
            xref="paper",
            yref="y",
            line=dict(dash="dot", width=1)
        )
        fig.add_annotation(
            x=1,
            y=1.5,
            xref="paper",
            yref="y",
            text="Severe threshold",
            showarrow=False,
            xanchor="right",
            yshift=10
        )

        fig.update_layout(
            title="SCFI trend and major disruption markers",
            xaxis_title="Month",
            yaxis_title="SCFI",
            height=485,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=55, b=20)
        )

        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("#### Automated briefing")

        insights = generate_insights(results, selected_model)
        for text in insights:
            st.markdown(
                f'<div class="insight-box">{text}</div>',
                unsafe_allow_html=True
            )

        st.caption(
            f"Latest common month: {current_row['Date']}"
        )


# =========================================================
# DATA QUALITY
# =========================================================

with pages[1]:
    st.markdown("## Data Quality & Harmonisation")

    st.dataframe(
        results["quality"],
        hide_index=True,
        width="stretch"
    )

    st.markdown("### Common analytical period")
    q1, q2, q3 = st.columns(3)
    q1.metric("Merged observations", f"{len(results['master']):,}")
    q2.metric("Start month", str(results["master"]["Date"].min()))
    q3.metric("End month", str(results["master"]["Date"].max()))

    with st.expander("Preview merged data"):
        preview = results["master"].tail(30).copy()
        preview["Date"] = preview["Date"].astype(str)
        st.dataframe(preview, hide_index=True, width="stretch")


# =========================================================
# SCFI ANALYSIS
# =========================================================

with pages[2]:
    st.markdown("## SCFI Development & Indicator Contribution")

    p1, p2 = st.columns([1, 1.3])

    with p1:
        pca_display = results["pca_table"].copy()
        pca_display["Weight (%)"] = pca_display["Weight (%)"].round(2)

        st.dataframe(
            pca_display[
                ["Indicator", "PC1 Loading", "Weight (%)"]
            ].round(4),
            hide_index=True,
            width="stretch"
        )

        formula = " + ".join(
            f"{weight:.3f}·{name}*"
            for name, weight in results["weights"].items()
        )
        st.info("SCFI = " + formula)

        st.metric(
            "PC1 explained variance",
            f"{results['pca'].explained_variance_ratio_[0] * 100:.2f}%"
        )

    with p2:
        contribution = results["pca_table"].sort_values(
            "Weight (%)",
            ascending=True
        )

        bar = px.bar(
            contribution,
            x="Weight (%)",
            y="Indicator",
            orientation="h",
            title="PCA-derived contribution to SCFI"
        )
        bar.update_layout(height=390)
        st.plotly_chart(bar, width="stretch")

    left, right = st.columns(2)

    with left:
        pie = px.pie(
            results["pca_table"],
            values="Weight (%)",
            names="Indicator",
            hole=.48,
            title="Relative indicator contribution"
        )
        pie.update_layout(height=420)
        st.plotly_chart(pie, width="stretch")

    with right:
        comparison = go.Figure()
        history = results["master_scaled"]

        comparison.add_trace(
            go.Scatter(
                x=history["Date"].astype(str),
                y=history["GSCPI"],
                name="GSCPI",
                mode="lines"
            )
        )
        comparison.add_trace(
            go.Scatter(
                x=history["Date"].astype(str),
                y=history["SCFI"],
                name="SCFI",
                mode="lines"
            )
        )
        comparison.update_layout(
            title="SCFI vs GSCPI",
            height=420,
            hovermode="x unified"
        )
        st.plotly_chart(comparison, width="stretch")

    st.markdown("### Validation")
    v1, v2, v3 = st.columns(3)
    v1.metric("Pearson correlation", f"{results['correlation']:.3f}")
    v2.metric("MAE vs GSCPI", f"{results['validation_mae']:.3f}")
    v3.metric("RMSE vs GSCPI", f"{results['validation_rmse']:.3f}")


# =========================================================
# EVENT EXPLORER
# =========================================================

with pages[3]:
    st.markdown("## Global Disruption Event Explorer")

    event_name = st.selectbox(
        "Select a disruption event",
        list(EVENTS.keys())
    )
    event_date = pd.Period(EVENTS[event_name], freq="M")

    ms = results["master_scaled"]
    window = ms[
        (ms["Date"] >= event_date - 6)
        & (ms["Date"] <= event_date + 12)
    ].copy()

    if window.empty:
        st.warning(
            "The selected event falls outside the common period of the uploaded data."
        )
    else:
        event_row = ms[ms["Date"] == event_date]

        e1, e2, e3 = st.columns(3)
        e1.metric("Event month", str(event_date))
        if not event_row.empty:
            event_scfi = float(event_row.iloc[0]["SCFI"])
            e2.metric("SCFI at event month", f"{event_scfi:.3f}")
            e3.metric(
                "Fragility class",
                event_row.iloc[0]["Fragility_Class"]
            )

        ef = go.Figure()

        ef.add_trace(
            go.Scatter(
                x=window["Date"].astype(str),
                y=window["SCFI"],
                mode="lines+markers",
                name="SCFI"
            )
        )
        ef.add_shape(
            type="line",
            x0=str(event_date),
            x1=str(event_date),
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(dash="dash", width=1)
        )
        ef.add_annotation(
            x=str(event_date),
            y=1,
            xref="x",
            yref="paper",
            text=event_name,
            showarrow=False,
            yshift=12
        )
        ef.update_layout(
            title=f"SCFI around {event_name}",
            xaxis_title="Month",
            yaxis_title="SCFI",
            height=450
        )

        st.plotly_chart(ef, width="stretch")

        st.caption(
            "The event explorer is contextual validation: it shows whether "
            "the index changed around known global disruptions; it does not "
            "claim that the event alone caused the SCFI movement."
        )


# =========================================================
# FORECASTING
# =========================================================

with pages[4]:
    st.markdown("## Forecasting & Model Comparison")

    st.caption(
        "The model selector affects the displayed one-month-ahead forecast. "
        "The dissertation-selected model remains Linear Regression because it "
        "achieved the best walk-forward continuous SCFI forecasting performance."
    )

    model_table = results["model_comparison"].copy()
    model_table[["MAE", "RMSE", "R²"]] = (
        model_table[["MAE", "RMSE", "R²"]].round(4)
    )

    st.dataframe(
        model_table,
        hide_index=True,
        width="stretch"
    )

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Selected model", selected_model)
    f2.metric("Forecast month", str(selected_forecast["Target"]))
    f3.metric("Forecast SCFI", f"{selected_forecast['Forecast']:.4f}")
    f4.metric("Forecast class", selected_forecast["Fragility"])

    wf = results["wf_results"][selected_model]["Predictions"]

    if not wf.empty:
        ff = go.Figure()
        ff.add_trace(
            go.Scatter(
                x=wf["Target_Date"].astype(str),
                y=wf["Actual_SCFI"],
                name="Actual SCFI",
                mode="lines"
            )
        )
        ff.add_trace(
            go.Scatter(
                x=wf["Target_Date"].astype(str),
                y=wf["Predicted_SCFI"],
                name=f"{selected_model} forecast",
                mode="lines",
                line=dict(dash="dash")
            )
        )
        ff.update_layout(
            title=f"Walk-forward performance: {selected_model}",
            xaxis_title="Forecast month",
            yaxis_title="SCFI",
            height=470,
            hovermode="x unified"
        )
        st.plotly_chart(ff, width="stretch")

        residual = px.histogram(
            wf,
            x="Residual",
            nbins=15,
            title=f"Residual distribution: {selected_model}"
        )
        residual.update_layout(height=350)
        st.plotly_chart(residual, width="stretch")

    recent = results["master_scaled"].tail(36)
    forecast_fig = go.Figure()

    forecast_fig.add_trace(
        go.Scatter(
            x=recent["Date"].astype(str),
            y=recent["SCFI"],
            name="Historical SCFI",
            mode="lines+markers"
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=[
                str(selected_forecast["Origin"]),
                str(selected_forecast["Target"])
            ],
            y=[
                current_scfi,
                selected_forecast["Forecast"]
            ],
            name="Next-month forecast",
            mode="lines+markers",
            line=dict(dash="dash", width=3)
        )
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=[
                str(selected_forecast["Target"]),
                str(selected_forecast["Target"])
            ],
            y=[
                selected_forecast["Lower"],
                selected_forecast["Upper"]
            ],
            mode="lines",
            name="Approx. 95% interval",
            line=dict(width=8)
        )
    )

    forecast_fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=.5,
        y1=.5,
        xref="paper",
        yref="y",
        line=dict(dash="dot", width=1)
    )
    forecast_fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=1.5,
        y1=1.5,
        xref="paper",
        yref="y",
        line=dict(dash="dot", width=1)
    )
    forecast_fig.update_layout(
        title="Final one-month-ahead SCFI outlook",
        xaxis_title="Month",
        yaxis_title="SCFI",
        height=480
    )
    st.plotly_chart(forecast_fig, width="stretch")


# =========================================================
# SCENARIO SIMULATOR
# =========================================================

with pages[5]:
    st.markdown("## What-if Scenario Simulator")
    st.caption(
        "Adjust the four raw indicators to estimate the SCFI implied by "
        "the current PCA weighting framework. This is a scenario tool, not "
        "a causal prediction model."
    )

    master = results["master"]
    latest = master.iloc[-1]

    s1, s2 = st.columns(2)

    with s1:
        sim_gscpi = st.slider(
            "GSCPI",
            float(master["GSCPI"].quantile(.01)),
            float(master["GSCPI"].quantile(.99)),
            float(np.clip(
                latest["GSCPI"],
                master["GSCPI"].quantile(.01),
                master["GSCPI"].quantile(.99)
            )),
            step=0.05
        )

        sim_inflation = st.slider(
            "Inflation rate (%)",
            float(master["Inflation_Rate"].quantile(.01)),
            float(master["Inflation_Rate"].quantile(.99)),
            float(np.clip(
                latest["Inflation_Rate"],
                master["Inflation_Rate"].quantile(.01),
                master["Inflation_Rate"].quantile(.99)
            )),
            step=0.10
        )

    with s2:
        sim_gpr = st.slider(
            "Geopolitical Risk Index",
            float(master["GPR"].quantile(.01)),
            float(master["GPR"].quantile(.99)),
            float(np.clip(
                latest["GPR"],
                master["GPR"].quantile(.01),
                master["GPR"].quantile(.99)
            )),
            step=1.0
        )

        sim_oil = st.slider(
            "Monthly oil return (%)",
            float(master["Oil_Return"].quantile(.01)),
            float(master["Oil_Return"].quantile(.99)),
            float(np.clip(
                latest["Oil_Return"],
                master["Oil_Return"].quantile(.01),
                master["Oil_Return"].quantile(.99)
            )),
            step=0.25
        )

    scenario_raw = pd.DataFrame([{
        "GSCPI": sim_gscpi,
        "Inflation_Rate": sim_inflation,
        "GPR": sim_gpr,
        "Oil_Return": sim_oil
    }])

    scenario_scaled = results["index_scaler"].transform(
        scenario_raw[VARIABLES]
    )[0]

    scenario_scfi = sum(
        results["weights"][name] * scenario_scaled[idx]
        for idx, name in enumerate(VARIABLES)
    )
    scenario_class = classify_fragility(scenario_scfi)

    a, b, c = st.columns(3)
    a.metric("Scenario SCFI", f"{scenario_scfi:.3f}")
    b.metric("Scenario fragility", scenario_class)
    c.metric(
        "Difference from current",
        f"{scenario_scfi - current_scfi:+.3f}"
    )

    st.markdown(
        risk_html(
            scenario_class,
            f"Scenario result: {scenario_class.upper()} fragility"
        ),
        unsafe_allow_html=True
    )

    scenario_table = pd.DataFrame({
        "Indicator": [
            "GSCPI",
            "Inflation Rate",
            "GPR",
            "Oil Return",
            "Scenario SCFI"
        ],
        "Scenario Value": [
            sim_gscpi,
            sim_inflation,
            sim_gpr,
            sim_oil,
            scenario_scfi
        ]
    })

    tornado = pd.DataFrame({
        "Indicator": VARIABLES,
        "PCA Weight (%)": [
            results["weights"][v] * 100
            for v in VARIABLES
        ]
    }).sort_values("PCA Weight (%)")

    tf = px.bar(
        tornado,
        x="PCA Weight (%)",
        y="Indicator",
        orientation="h",
        title="Structural sensitivity: PCA weight of each indicator"
    )
    tf.update_layout(height=380)
    st.plotly_chart(tf, width="stretch")


# =========================================================
# REPORTS
# =========================================================

with pages[6]:
    st.markdown("## Reports, Downloads & Method")

    insights = generate_insights(results, selected_model)

    st.markdown("### Executive interpretation")
    for line in insights:
        st.markdown(
            f'<div class="insight-box">{line}</div>',
            unsafe_allow_html=True
        )

    excel_bytes = create_excel_report(
        results,
        selected_model,
        scenario_table=scenario_table
    )

    pdf_bytes = create_pdf_report(
        results,
        selected_model,
        insights
    )

    d1, d2, d3 = st.columns(3)

    d1.download_button(
        "Download Excel analysis",
        data=excel_bytes,
        file_name="SCFI_Executive_Analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch"
    )

    d2.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name="SCFI_Executive_Report.pdf",
        mime="application/pdf",
        width="stretch"
    )

    scfi_csv = results["master_scaled"][
        ["Date", "SCFI", "Fragility_Class"]
    ].copy()
    scfi_csv["Date"] = scfi_csv["Date"].astype(str)

    d3.download_button(
        "Download SCFI history",
        data=scfi_csv.to_csv(index=False).encode("utf-8"),
        file_name="SCFI_History.csv",
        mime="text/csv",
        width="stretch"
    )

    st.markdown("### Methodology summary")
    st.markdown(
        """
        **Index construction**
        - Monthly GSCPI, inflation, GPR and WTI oil return are aligned.
        - Indicators are standardised using z-scores.
        - PCA is fitted to the four standardised indicators.
        - Absolute first-component loadings are normalised to sum to one.
        - SCFI is the weighted sum of the four standardised indicators.

        **Forecasting**
        - Six consecutive monthly lags are created for GSCPI, inflation,
          GPR, oil return and SCFI.
        - Models are evaluated using annual expanding walk-forward validation.
        - Linear Regression, Random Forest and XGBoost are compared.
        - The selected model is retrained on all available historical
          regression observations to produce a one-month-ahead SCFI forecast.
        """
    )

    st.warning(
        "The scenario simulator changes indicator inputs within the fitted "
        "PCA framework. It should be interpreted as structured what-if "
        "analysis rather than evidence that changing one indicator will "
        "causally produce the displayed SCFI."
    )

st.markdown(
    """
    <div class="rgu-footer">
      <strong>Supply Chain Fragility Monitor</strong> · MSc Business Analytics research application ·
      Robert Gordon University, Aberdeen
      <br>
      Academic prototype for analytical decision support. RGU is identified as the
      dissertation's academic institution; this application is not an official University service.
    </div>
    """,
    unsafe_allow_html=True
)

