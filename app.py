
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
# VERSION 3 INTERFACE OVERRIDES
# =========================================================

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: .65rem; max-width: 1550px; }}
      [data-testid="stHeader"] {{ background: rgba(245,246,248,.96); border-bottom:1px solid #e6dfe9; }}
      .v3-topbar {{
        background: linear-gradient(100deg,{RGU_PURPLE_DEEP} 0%,{RGU_PURPLE_DARK} 58%,{RGU_PURPLE} 100%);
        border-radius: 14px; border-bottom: 5px solid {RGU_CYAN};
        padding: 20px 24px; margin: 0 0 20px 0; display:flex; align-items:center;
        justify-content:space-between; gap:24px; box-shadow:0 8px 22px rgba(70,25,80,.10);
      }}
      .v3-brand {{ display:flex; align-items:center; gap:18px; }}
      .v3-mark {{ min-width:110px; border:2px solid rgba(255,255,255,.78); border-radius:8px;
        color:#fff; font-weight:900; line-height:1.0; text-align:center; padding:9px 10px; letter-spacing:.02em; }}
      .v3-title h1 {{ color:#fff !important; margin:0; font-size:2rem; }}
      .v3-title p {{ color:rgba(255,255,255,.80); margin:5px 0 0 0; font-size:.92rem; }}
      .v3-meta {{ color:#fff; text-align:right; min-width:180px; }}
      .v3-meta small {{ color:rgba(255,255,255,.72); }}
      .v3-page h2 {{ color:{RGU_PURPLE_DARK}!important; margin:.1rem 0 .1rem 0; font-size:1.8rem; }}
      .v3-page p {{ color:{RGU_MUTED}; margin:.1rem 0 1rem 0; }}
      .nav-brand {{ padding:8px 0 13px 0; border-bottom:1px solid rgba(255,255,255,.18); margin-bottom:10px; }}
      .nav-brand .big {{ color:#fff; font-size:1.1rem; font-weight:900; }}
      .nav-brand .small {{ color:rgba(255,255,255,.72); font-size:.78rem; margin-top:4px; }}
      .timeline-item {{ background:#fff; border:1px solid {RGU_BORDER}; border-left:4px solid {RGU_PURPLE};
        border-radius:8px; padding:10px 12px; margin-bottom:9px; }}
      .timeline-date {{ color:{RGU_MUTED}; font-size:.74rem; font-weight:700; }}
      .timeline-title {{ color:{RGU_PURPLE_DARK}; font-weight:800; margin:2px 0; }}
      .footer {{ border-top:3px solid {RGU_PURPLE}; margin-top:30px; padding-top:12px; color:{RGU_MUTED}; font-size:.79rem; }}
      @media(max-width:1000px){{ .v3-topbar{{flex-direction:column;align-items:flex-start;}} .v3-meta{{text-align:left;}} }}
    </style>
    """,
    unsafe_allow_html=True,
)

if "results" not in st.session_state:
    st.session_state.results = None

# ------------------------- Sidebar navigation -------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="nav-brand">
          <div class="big">SCFI Monitor</div>
          <div class="small">MSc Business Analytics · RGU</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        [
            "Executive Dashboard",
            "Data Workspace",
            "Data Validation",
            "SCFI Analysis",
            "Forecasting",
            "Scenario Simulator",
            "Global Events",
            "Reports & Downloads",
            "Methodology",
            "About This App",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    if st.session_state.results is not None:
        _latest = st.session_state.results["master_scaled"].iloc[-1]
        st.success("All datasets analysed")
        st.caption(f"Latest month: {_latest['Date']}")
        st.caption(f"SCFI: {_latest['SCFI']:.3f} · {_latest['Fragility_Class']}")
    else:
        st.info("Upload and analyse data in Data Workspace.")
    st.divider()
    st.caption("Academic research application")
    st.caption("Not an official RGU service")

# ------------------------- Fixed top header -------------------------
st.markdown(
    f"""
    <div class="v3-topbar">
      <div class="v3-brand">
        <div class="v3-mark">RGU<br>ROBERT<br>GORDON</div>
        <div class="v3-title">
          <h1>Supply Chain Fragility Monitor</h1>
          <p>MSc Business Analytics Dissertation · Robert Gordon University, Aberdeen</p>
        </div>
      </div>
      <div class="v3-meta"><strong>{datetime.now().strftime('%B %Y')}</strong><br><small>Decision-support research application</small></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Data workspace -------------------------
if page == "Data Workspace":
    st.markdown('<div class="v3-page"><h2>Data Workspace</h2><p>Upload, map and validate the four source datasets used by the SCFI framework.</p></div>', unsafe_allow_html=True)
    st.markdown("### 1 · Upload datasets")
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    with c1:
        gscpi_file = st.file_uploader("GSCPI", type=["csv","xls","xlsx"], key="v3_gscpi")
    with c2:
        inflation_file = st.file_uploader("Inflation", type=["csv","xls","xlsx"], key="v3_inflation")
    with c3:
        gpr_file = st.file_uploader("Geopolitical Risk (GPR)", type=["csv","xls","xlsx"], key="v3_gpr")
    with c4:
        oil_file = st.file_uploader("WTI Oil", type=["csv","xls","xlsx"], key="v3_oil")

    ready = all([gscpi_file, inflation_file, gpr_file, oil_file])
    if not ready:
        st.info("Upload all four files to continue to column mapping.")
    else:
        st.markdown("### 2 · Configure uploaded files")
        tabs = st.tabs(["GSCPI","Inflation","GPR","Oil"])

        def configure(uploaded, prefix, suggested_date, suggested_value, default_skip=0, preferred_sheet=None):
            sheets = excel_sheet_names(uploaded)
            sheet = None
            if sheets:
                idx = sheets.index(preferred_sheet) if preferred_sheet in sheets else 0
                sheet = st.selectbox("Worksheet", sheets, index=idx, key=f"{prefix}_sheet")
            raw = read_uploaded_file(uploaded, sheet)
            cols = list(raw.columns)
            di = cols.index(suggested_date) if suggested_date in cols else 0
            vi = cols.index(suggested_value) if suggested_value in cols else min(1, len(cols)-1)
            a,b,c = st.columns(3)
            with a:
                date_col = st.selectbox("Date column", cols, index=di, key=f"{prefix}_date")
            with b:
                value_col = st.selectbox("Value column", cols, index=vi, key=f"{prefix}_value")
            with c:
                skip = st.number_input("Rows to skip before data starts", 0, max(0,len(raw)-1), min(default_skip,max(0,len(raw)-1)), 1, key=f"{prefix}_skip")
            st.caption(f"Detected {len(raw):,} rows × {len(cols)} columns.")
            st.dataframe(raw.head(8), width="stretch")
            return raw,date_col,value_col,skip

        with tabs[0]:
            g_raw,g_date,g_value,g_skip = configure(gscpi_file,"gcfg","Date","GSCPI",4,"GSCPI Monthly Data")
        with tabs[1]:
            i_raw,i_date,i_value,i_skip = configure(inflation_file,"icfg","TIME_PERIOD","OBS_VALUE")
        with tabs[2]:
            gp_raw,gp_date,gp_value,gp_skip = configure(gpr_file,"gpcfg","month","GPR",0,"Sheet1")
        with tabs[3]:
            o_raw,o_date,o_value,o_skip = configure(oil_file,"ocfg","observation_date","DCOILWTICO")
            oil_mode = st.radio("Uploaded oil value", ["Oil price — calculate monthly return","Monthly oil return already calculated"], key="v3_oilmode")

        st.markdown("### 3 · Build SCFI")
        if st.button("Validate data and build SCFI", type="primary", width="stretch"):
            try:
                with st.spinner("Cleaning datasets, constructing SCFI and evaluating forecasting models..."):
                    gscpi = clean_two_column_data(g_raw,g_date,g_value,"GSCPI",g_skip)
                    inflation = clean_two_column_data(i_raw,i_date,i_value,"Inflation_Rate",i_skip)
                    gpr = clean_two_column_data(gp_raw,gp_date,gp_value,"GPR",gp_skip)
                    oil = clean_oil_data(o_raw,o_date,o_value,oil_mode,o_skip)
                    st.session_state.results = analyse_uploaded_data(gscpi,inflation,gpr,oil)
                st.success("SCFI analysis completed successfully.")
            except Exception as exc:
                st.error("The uploaded data could not be analysed.")
                st.exception(exc)

# ------------------------- Empty state -------------------------
results = st.session_state.results
if results is None and page != "Data Workspace":
    st.markdown('<div class="v3-page"><h2>Supply Chain Fragility Monitor</h2><p>No analytical dataset is active yet.</p></div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.info("**Upload data**\n\nOpen Data Workspace and provide the four source datasets.")
    b.info("**Build SCFI**\n\nMap the fields and run the calculation workflow.")
    c.info("**Explore results**\n\nReturn to the Executive Dashboard after analysis completes.")
    st.stop()

if results is not None:
    current = results["master_scaled"].iloc[-1]
    previous = results["master_scaled"].iloc[-2]
    latest_raw = results["master"].iloc[-1]
    previous_raw = results["master"].iloc[-2]
    models = results["model_comparison"]["Model"].tolist()
    selected_model = st.sidebar.selectbox("Forecast model", models, index=models.index("Linear Regression") if "Linear Regression" in models else 0)
    selected_forecast = results["final_forecasts"][selected_model]
    current_scfi = float(current["SCFI"])

# ------------------------- Executive dashboard -------------------------
if page == "Executive Dashboard":
    st.markdown('<div class="v3-page"><h2>Executive Dashboard</h2><p>Current fragility conditions, major drivers and next-month outlook.</p></div>', unsafe_allow_html=True)
    r1 = st.columns(3); r2 = st.columns(3)
    r1[0].metric("Current SCFI", f"{current_scfi:.4f}", delta=f"{current_scfi-float(previous['SCFI']):+.3f}")
    r1[1].metric("Current GSCPI", f"{latest_raw['GSCPI']:.4f}", delta=f"{latest_raw['GSCPI']-previous_raw['GSCPI']:+.3f}")
    r1[2].metric("Inflation Rate", f"{latest_raw['Inflation_Rate']:.4f}", delta=f"{latest_raw['Inflation_Rate']-previous_raw['Inflation_Rate']:+.3f}")
    r2[0].metric("Geopolitical Risk", f"{latest_raw['GPR']:.4f}", delta=f"{latest_raw['GPR']-previous_raw['GPR']:+.3f}")
    r2[1].metric("Oil Return", f"{latest_raw['Oil_Return']:.3f}%", delta=f"{latest_raw['Oil_Return']-previous_raw['Oil_Return']:+.3f}")
    r2[2].metric("Fragility Level", current["Fragility_Class"])

    left,right = st.columns([2.2,1])
    with left:
        history = results["master_scaled"].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["Date"].astype(str), y=history["SCFI"], mode="lines", name="SCFI", line=dict(color=RGU_PURPLE,width=3)))
        fig.add_trace(go.Scatter(x=history["Date"].astype(str), y=history["SCFI"].rolling(3).mean(), mode="lines", name="3-month MA", line=dict(color=RGU_CYAN,dash="dash",width=2)))
        for event,date in EVENTS.items():
            if date in history["Date"].astype(str).values:
                fig.add_shape(type="line",x0=date,x1=date,y0=0,y1=1,xref="x",yref="paper",line=dict(color=RGU_PURPLE,dash="dash",width=1))
                fig.add_annotation(x=date,y=1,xref="x",yref="paper",text=event,showarrow=False,textangle=-35,yshift=12,font=dict(size=10,color=RGU_PURPLE_DARK))
        for y,color,label in [(0.5,RGU_GOLD,"High threshold"),(1.5,"#C84B45","Severe threshold")]:
            fig.add_shape(type="line",x0=0,x1=1,y0=y,y1=y,xref="paper",yref="y",line=dict(color=color,dash="dot",width=1))
            fig.add_annotation(x=1,y=y,xref="paper",yref="y",text=label,showarrow=False,xanchor="right",yshift=10,font=dict(size=10,color=color))
        fig.update_layout(title="SCFI Trend Over Time",height=480,hovermode="x unified",xaxis_title="Month",yaxis_title="SCFI")
        st.plotly_chart(fig,width="stretch")
    with right:
        st.markdown("### Recent World Events")
        for event,date in list(EVENTS.items())[::-1]:
            period = pd.Period(date,freq="M")
            row = results["master_scaled"].loc[results["master_scaled"]["Date"]==period]
            value = f"SCFI {float(row.iloc[0]['SCFI']):.2f}" if not row.empty else "Outside common period"
            st.markdown(f'<div class="timeline-item"><div class="timeline-date">{date}</div><div class="timeline-title">{event}</div><div class="muted">{value}</div></div>',unsafe_allow_html=True)

    b1,b2,b3 = st.columns([1,1.25,1.4])
    with b1:
        donut = px.pie(results["pca_table"],values="Weight (%)",names="Indicator",hole=.58,title="PCA Contribution to SCFI")
        donut.update_layout(height=410)
        st.plotly_chart(donut,width="stretch")
    with b2:
        recent = results["master_scaled"].tail(30)
        ff = go.Figure()
        ff.add_trace(go.Scatter(x=recent["Date"].astype(str),y=recent["SCFI"],mode="lines",name="Historical",line=dict(color=RGU_PURPLE,width=3)))
        ff.add_trace(go.Scatter(x=[str(selected_forecast["Origin"]),str(selected_forecast["Target"])],y=[current_scfi,selected_forecast["Forecast"]],mode="lines+markers",name="Forecast",line=dict(color=RGU_CYAN,width=3,dash="dash")))
        ff.update_layout(title="Forecast · Next Month",height=410,xaxis_title="Month",yaxis_title="SCFI")
        st.plotly_chart(ff,width="stretch")
        st.info(f"{selected_forecast['Target']} · SCFI {selected_forecast['Forecast']:.3f} · {selected_forecast['Fragility']}")
    with b3:
        st.markdown("### Automated Briefing")
        for line in generate_insights(results,selected_model):
            st.markdown(f'<div class="insight-box">{line}</div>',unsafe_allow_html=True)
        st.caption(f"Selected model: {selected_model}")

elif page == "Data Validation":
    st.markdown('<div class="v3-page"><h2>Data Validation</h2><p>Coverage, missingness and the common analytical period.</p></div>',unsafe_allow_html=True)
    st.dataframe(results["quality"],width="stretch",hide_index=True)
    a,b,c = st.columns(3)
    a.metric("Merged observations",f"{len(results['master']):,}")
    b.metric("Start month",str(results["master"]["Date"].min()))
    c.metric("End month",str(results["master"]["Date"].max()))
    prev = results["master"].tail(24).copy(); prev["Date"] = prev["Date"].astype(str)
    st.dataframe(prev,width="stretch",hide_index=True)

elif page == "SCFI Analysis":
    st.markdown('<div class="v3-page"><h2>SCFI Analysis</h2><p>PCA-derived weights, index construction and validation against GSCPI.</p></div>',unsafe_allow_html=True)
    a,b = st.columns([1,1.6])
    with a:
        display = results["pca_table"].copy(); display["Weight (%)"] = display["Weight (%)"].round(2)
        st.dataframe(display[["Indicator","PC1 Loading","Weight (%)"]].round(4),width="stretch",hide_index=True)
        formula = " + ".join(f"{w:.3f}·{n}*" for n,w in results["weights"].items())
        st.info("SCFI = "+formula)
        st.metric("PC1 explained variance",f"{results['pca'].explained_variance_ratio_[0]*100:.2f}%")
    with b:
        contrib = results["pca_table"].sort_values("Weight (%)")
        bar = px.bar(contrib,x="Weight (%)",y="Indicator",orientation="h",title="Indicator Contribution to SCFI")
        bar.update_layout(height=410); st.plotly_chart(bar,width="stretch")
    comparison = go.Figure()
    comparison.add_trace(go.Scatter(x=results["master_scaled"]["Date"].astype(str),y=results["master_scaled"]["SCFI"],mode="lines",name="SCFI",line=dict(color=RGU_PURPLE,width=3)))
    comparison.add_trace(go.Scatter(x=results["master_scaled"]["Date"].astype(str),y=results["master_scaled"]["GSCPI"],mode="lines",name="GSCPI",line=dict(color=RGU_CYAN,width=2)))
    comparison.update_layout(title="SCFI vs GSCPI",height=450,hovermode="x unified"); st.plotly_chart(comparison,width="stretch")
    v1,v2,v3 = st.columns(3); v1.metric("Pearson correlation",f"{results['correlation']:.3f}"); v2.metric("MAE vs GSCPI",f"{results['validation_mae']:.3f}"); v3.metric("RMSE vs GSCPI",f"{results['validation_rmse']:.3f}")

elif page == "Forecasting":
    st.markdown('<div class="v3-page"><h2>Forecasting</h2><p>Walk-forward model comparison and one-month-ahead SCFI outlook.</p></div>',unsafe_allow_html=True)
    mt = results["model_comparison"].copy(); mt[["MAE","RMSE","R²"]] = mt[["MAE","RMSE","R²"]].round(4)
    st.dataframe(mt,width="stretch",hide_index=True)
    a,b,c,d = st.columns(4); a.metric("Selected model",selected_model); b.metric("Forecast month",str(selected_forecast["Target"])); c.metric("Forecast SCFI",f"{selected_forecast['Forecast']:.4f}"); d.metric("Forecast fragility",selected_forecast["Fragility"])
    wf = results["wf_results"][selected_model]["Predictions"]
    if not wf.empty:
        fig = go.Figure(); fig.add_trace(go.Scatter(x=wf["Target_Date"].astype(str),y=wf["Actual_SCFI"],mode="lines",name="Actual",line=dict(color=RGU_PURPLE,width=3))); fig.add_trace(go.Scatter(x=wf["Target_Date"].astype(str),y=wf["Predicted_SCFI"],mode="lines",name="Predicted",line=dict(color=RGU_CYAN,width=2,dash="dash"))); fig.update_layout(title=f"Walk-Forward Performance · {selected_model}",height=470,hovermode="x unified"); st.plotly_chart(fig,width="stretch")

elif page == "Scenario Simulator":
    st.markdown('<div class="v3-page"><h2>Scenario Simulator</h2><p>Explore how different indicator combinations map into the fitted SCFI framework.</p></div>',unsafe_allow_html=True)
    master=results["master"]; latest=master.iloc[-1]; c1,c2=st.columns(2)
    with c1:
        sg=st.slider("GSCPI",float(master.GSCPI.quantile(.01)),float(master.GSCPI.quantile(.99)),float(np.clip(latest.GSCPI,master.GSCPI.quantile(.01),master.GSCPI.quantile(.99))),step=.05)
        si=st.slider("Inflation rate",float(master.Inflation_Rate.quantile(.01)),float(master.Inflation_Rate.quantile(.99)),float(np.clip(latest.Inflation_Rate,master.Inflation_Rate.quantile(.01),master.Inflation_Rate.quantile(.99))),step=.1)
    with c2:
        sgpr=st.slider("Geopolitical Risk Index",float(master.GPR.quantile(.01)),float(master.GPR.quantile(.99)),float(np.clip(latest.GPR,master.GPR.quantile(.01),master.GPR.quantile(.99))),step=1.0)
        so=st.slider("Monthly oil return (%)",float(master.Oil_Return.quantile(.01)),float(master.Oil_Return.quantile(.99)),float(np.clip(latest.Oil_Return,master.Oil_Return.quantile(.01),master.Oil_Return.quantile(.99))),step=.25)
    raw=pd.DataFrame([{"GSCPI":sg,"Inflation_Rate":si,"GPR":sgpr,"Oil_Return":so}]); scaled=results["index_scaler"].transform(raw[VARIABLES])[0]; scenario=sum(results["weights"][n]*scaled[i] for i,n in enumerate(VARIABLES)); label=classify_fragility(scenario)
    a,b,c=st.columns(3); a.metric("Scenario SCFI",f"{scenario:.3f}"); b.metric("Scenario fragility",label); c.metric("Change vs current",f"{scenario-current_scfi:+.3f}"); st.warning("This is structured what-if analysis within the fitted PCA framework, not causal forecasting.")

elif page == "Global Events":
    st.markdown('<div class="v3-page"><h2>Global Events</h2><p>Contextual exploration of SCFI around major supply-chain disruptions.</p></div>',unsafe_allow_html=True)
    event=st.selectbox("Select event",list(EVENTS.keys())); period=pd.Period(EVENTS[event],freq="M"); hist=results["master_scaled"]; window=hist[(hist["Date"]>=period-6)&(hist["Date"]<=period+12)].copy(); row=hist[hist["Date"]==period]
    a,b,c=st.columns(3); a.metric("Event month",str(period));
    if not row.empty: b.metric("SCFI at event",f"{float(row.iloc[0]['SCFI']):.3f}"); c.metric("Fragility class",row.iloc[0]["Fragility_Class"])
    fig=go.Figure(); fig.add_trace(go.Scatter(x=window["Date"].astype(str),y=window["SCFI"],mode="lines+markers",name="SCFI",line=dict(color=RGU_PURPLE,width=3))); fig.add_shape(type="line",x0=str(period),x1=str(period),y0=0,y1=1,xref="x",yref="paper",line=dict(color=RGU_PURPLE,dash="dash",width=1)); fig.add_annotation(x=str(period),y=1,xref="x",yref="paper",text=event,showarrow=False,yshift=12); fig.update_layout(title=f"SCFI Around {event}",height=450,xaxis_title="Month",yaxis_title="SCFI"); st.plotly_chart(fig,width="stretch"); st.caption("Contextual validation only; this does not claim the event alone caused the SCFI movement.")

elif page == "Reports & Downloads":
    st.markdown('<div class="v3-page"><h2>Reports & Downloads</h2><p>Export analytical outputs for dissertation evidence and management review.</p></div>',unsafe_allow_html=True)
    insights=generate_insights(results,selected_model); excel_bytes=create_excel_report(results,selected_model,None); pdf_bytes=create_pdf_report(results,selected_model,insights); csv=results["master_scaled"][["Date","SCFI","Fragility_Class"]].copy(); csv["Date"]=csv["Date"].astype(str)
    a,b,c=st.columns(3); a.download_button("Download Excel analysis",excel_bytes,"SCFI_Executive_Analysis.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",width="stretch"); b.download_button("Download PDF report",pdf_bytes,"SCFI_Executive_Report.pdf","application/pdf",width="stretch"); c.download_button("Download SCFI history",csv.to_csv(index=False).encode(),"SCFI_History.csv","text/csv",width="stretch")
    st.markdown("### Executive interpretation")
    for line in insights: st.markdown(f'<div class="insight-box">{line}</div>',unsafe_allow_html=True)

elif page == "Methodology":
    st.markdown('<div class="v3-page"><h2>Methodology</h2><p>Technical summary of the SCFI construction and forecasting workflow.</p></div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        st.markdown("""### Index construction\n1. Harmonise GSCPI, inflation, GPR and WTI oil data to monthly frequency.\n2. Convert daily oil prices into monthly returns.\n3. Standardise the four indicators with z-scores.\n4. Fit PCA to the standardised variables.\n5. Normalise absolute PC1 loadings to sum to one.\n6. Calculate SCFI as the weighted sum of the four standardised indicators.\n\n### Fragility thresholds\n- **Moderate:** SCFI < 0.5\n- **High:** 0.5 ≤ SCFI < 1.5\n- **Severe:** SCFI ≥ 1.5""")
    with b:
        st.markdown("""### Forecasting framework\n- Six consecutive monthly lags for GSCPI, inflation, GPR, oil return and SCFI.\n- Expanding annual walk-forward validation.\n- Linear Regression, Random Forest and XGBoost comparison.\n- Selected model retrained on all usable history.\n- Genuine one-month-ahead SCFI forecast.\n\n### Validation\n- Pearson correlation against GSCPI\n- MAE and RMSE\n- Historical event comparison\n- Walk-forward regression evaluation""")

elif page == "About This App":
    st.markdown('<div class="v3-page"><h2>About This App</h2><p>Research application developed to operationalise the Supply Chain Fragility Index.</p></div>',unsafe_allow_html=True)
    st.markdown("""The **Supply Chain Fragility Monitor** is an MSc Business Analytics research application developed around the Supply Chain Fragility Index (SCFI).\n\nIt integrates operational, macroeconomic, geopolitical and energy-market signals to provide current fragility monitoring, PCA contribution analysis, historical event exploration, walk-forward model comparison, one-month-ahead forecasting, scenario analysis and downloadable reports.\n\n**Academic note:** Robert Gordon University is identified as the academic institution associated with the dissertation. This application is not an official University system or service.""")

st.markdown('<div class="footer"><strong>Supply Chain Fragility Monitor</strong> · MSc Business Analytics research application · Robert Gordon University, Aberdeen<br>Academic prototype for analytical decision support. Not an official University service.</div>',unsafe_allow_html=True)
