
import io
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SCFI Development and Forecasting Dashboard",
    page_icon="📦",
    layout="wide"
)

st.title("Supply Chain Fragility Index Dashboard")
st.caption(
    "Upload GSCPI, inflation, geopolitical-risk and oil datasets; "
    "calculate the PCA-weighted SCFI; validate the index; forecast "
    "next-month fragility; and download the analytical results."
)

VARIABLES = ["GSCPI", "Inflation_Rate", "GPR", "Oil_Return"]
FEATURE_COLUMNS = VARIABLES + ["SCFI"]
LAGS = [1, 2, 3, 4, 5, 6]


# =========================================================
# HELPERS
# =========================================================

def read_uploaded_file(uploaded_file, sheet_name=None):
    """Read CSV, XLS or XLSX uploads without writing them to disk."""
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)

    if name.endswith((".xls", ".xlsx")):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)

    raise ValueError("Unsupported file type. Upload CSV, XLS or XLSX.")


def excel_sheet_names(uploaded_file):
    name = uploaded_file.name.lower()

    if not name.endswith((".xls", ".xlsx")):
        return []

    uploaded_file.seek(0)
    workbook = pd.ExcelFile(uploaded_file)
    return workbook.sheet_names


def clean_two_column_data(
    raw,
    date_column,
    value_column,
    output_name,
    start_row=0
):
    """Standard cleaning used for GSCPI, inflation and GPR."""
    data = raw.iloc[int(start_row):][
        [date_column, value_column]
    ].copy()

    data.columns = ["Date", output_name]

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data[output_name] = pd.to_numeric(
        data[output_name],
        errors="coerce"
    )

    data = (
        data
        .dropna(subset=["Date", output_name])
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    data["Date"] = data["Date"].dt.to_period("M")

    # If multiple rows exist in one month, retain the monthly mean.
    data = (
        data
        .groupby("Date", as_index=False)[output_name]
        .mean()
    )

    return data


def clean_oil_data(
    raw,
    date_column,
    value_column,
    uploaded_value_type
):
    """
    Convert uploaded oil prices into monthly returns, or accept already
    calculated monthly returns.
    """
    oil = raw[[date_column, value_column]].copy()
    oil.columns = ["Date", "Oil_Value"]

    oil["Date"] = pd.to_datetime(
        oil["Date"],
        errors="coerce"
    )

    oil["Oil_Value"] = pd.to_numeric(
        oil["Oil_Value"],
        errors="coerce"
    )

    oil = (
        oil
        .dropna(subset=["Date", "Oil_Value"])
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if uploaded_value_type == "Oil price — calculate monthly return":
        oil_monthly = (
            oil
            .set_index("Date")
            .resample("ME")
            .mean()
            .reset_index()
        )

        oil_monthly["Oil_Return"] = (
            oil_monthly["Oil_Value"]
            .pct_change()
            .mul(100)
        )

        result = (
            oil_monthly[["Date", "Oil_Return"]]
            .dropna()
            .copy()
        )
    else:
        result = oil.rename(
            columns={"Oil_Value": "Oil_Return"}
        )[["Date", "Oil_Return"]]

        result = (
            result
            .set_index("Date")
            .resample("ME")
            .mean()
            .reset_index()
        )

    result["Date"] = result["Date"].dt.to_period("M")

    return result


def classify_fragility(scfi_value):
    if scfi_value < 0.5:
        return "Moderate"
    if scfi_value < 1.5:
        return "High"
    return "Severe"


def dataframe_to_excel(results):
    """Create a downloadable Excel workbook in memory."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results["data_quality"].to_excel(
            writer,
            sheet_name="Data Quality",
            index=False
        )
        results["merged_data"].to_excel(
            writer,
            sheet_name="Merged Data",
            index=False
        )
        results["standardised_data"].to_excel(
            writer,
            sheet_name="Standardised Data",
            index=False
        )
        results["pca_table"].to_excel(
            writer,
            sheet_name="PCA Weights",
            index=False
        )
        results["scfi_results"].to_excel(
            writer,
            sheet_name="SCFI Results",
            index=False
        )
        results["validation_table"].to_excel(
            writer,
            sheet_name="Validation",
            index=False
        )
        results["forecast_table"].to_excel(
            writer,
            sheet_name="Forecast",
            index=False
        )

        if not results["walk_forward_predictions"].empty:
            results["walk_forward_predictions"].to_excel(
                writer,
                sheet_name="Walk Forward Forecasts",
                index=False
            )

    output.seek(0)
    return output.getvalue()


def create_html_report(results):
    """Create a compact downloadable analytical report."""
    weights_html = results["pca_table"].to_html(
        index=False,
        border=0
    )

    validation_html = results["validation_table"].to_html(
        index=False,
        border=0
    )

    forecast_html = results["forecast_table"].to_html(
        index=False,
        border=0
    )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>SCFI Analytical Report</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 40px;
          color: #222;
          line-height: 1.5;
        }}
        h1, h2 {{ color: #173b6c; }}
        table {{
          border-collapse: collapse;
          width: 100%;
          margin-bottom: 24px;
        }}
        th, td {{
          border: 1px solid #d6dce5;
          padding: 8px;
          text-align: left;
        }}
        th {{ background: #eef3f9; }}
        .note {{
          background: #fff7df;
          border-left: 5px solid #d39b18;
          padding: 12px;
        }}
      </style>
    </head>
    <body>
      <h1>Supply Chain Fragility Index Analytical Report</h1>
      <p><strong>Generated:</strong> {generated_at}</p>

      <h2>Dataset summary</h2>
      <p>
        The uploaded datasets were harmonised to monthly frequency and merged
        using the common observation period. The resulting analytical dataset
        contained <strong>{len(results["merged_data"])}</strong> observations,
        covering <strong>{results["merged_data"]["Date"].min()}</strong> to
        <strong>{results["merged_data"]["Date"].max()}</strong>.
      </p>

      <h2>PCA-derived SCFI weights</h2>
      {weights_html}

      <h2>Validation results</h2>
      {validation_html}

      <h2>One-month-ahead forecast</h2>
      {forecast_html}

      <div class="note">
        Forecasts are analytical decision-support estimates and should not be
        interpreted as certainty. The model depends on the quality, timeliness
        and comparability of the uploaded data.
      </div>
    </body>
    </html>
    """

    return html.encode("utf-8")


def calculate_scfi_and_forecast(
    gscpi,
    inflation,
    gpr,
    oil
):
    # -------------------------
    # Data-quality summary
    # -------------------------
    quality_rows = []

    for name, frame, value_col in [
        ("GSCPI", gscpi, "GSCPI"),
        ("Inflation", inflation, "Inflation_Rate"),
        ("GPR", gpr, "GPR"),
        ("Oil Return", oil, "Oil_Return")
    ]:
        quality_rows.append({
            "Dataset": name,
            "Rows after cleaning": len(frame),
            "Start month": str(frame["Date"].min()),
            "End month": str(frame["Date"].max()),
            "Missing values": int(
                frame[value_col].isna().sum()
            )
        })

    data_quality = pd.DataFrame(quality_rows)

    # -------------------------
    # Merge common months
    # -------------------------
    master = (
        gscpi
        .merge(inflation, on="Date", how="inner")
        .merge(gpr, on="Date", how="inner")
        .merge(oil, on="Date", how="inner")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(master) < 24:
        raise ValueError(
            "The common merged period contains fewer than 24 monthly "
            "observations. More overlapping data are required."
        )

    # -------------------------
    # Standardisation
    # -------------------------
    index_scaler = StandardScaler()
    scaled_matrix = index_scaler.fit_transform(
        master[VARIABLES]
    )

    master_scaled = master[["Date"]].copy()
    master_scaled[VARIABLES] = scaled_matrix

    # -------------------------
    # PCA and weights
    # -------------------------
    pca = PCA()
    pca.fit(master_scaled[VARIABLES])

    pc1_loadings = pca.components_[0]
    absolute_loadings = np.abs(pc1_loadings)
    normalised_weights = (
        absolute_loadings / absolute_loadings.sum()
    )

    pca_table = pd.DataFrame({
        "Variable": VARIABLES,
        "PC1 Loading": pc1_loadings,
        "Absolute Loading": absolute_loadings,
        "Normalised Weight": normalised_weights,
        "Weight (%)": normalised_weights * 100
    })

    # Same final index construction used in the notebook:
    # normalised absolute PC1 loadings.
    master_scaled["SCFI"] = sum(
        normalised_weights[index] * master_scaled[column]
        for index, column in enumerate(VARIABLES)
    )

    master_scaled["Fragility_Class"] = (
        master_scaled["SCFI"].apply(classify_fragility)
    )

    # -------------------------
    # Validation against GSCPI
    # -------------------------
    correlation = master_scaled[
        ["SCFI", "GSCPI"]
    ].corr().iloc[0, 1]

    validation_rmse = np.sqrt(
        mean_squared_error(
            master_scaled["GSCPI"],
            master_scaled["SCFI"]
        )
    )

    validation_mae = mean_absolute_error(
        master_scaled["GSCPI"],
        master_scaled["SCFI"]
    )

    validation_table = pd.DataFrame({
        "Metric": [
            "Pearson correlation",
            "RMSE",
            "MAE",
            "PC1 explained variance"
        ],
        "Value": [
            correlation,
            validation_rmse,
            validation_mae,
            pca.explained_variance_ratio_[0]
        ]
    })

    # -------------------------
    # Six-month lag engineering
    # -------------------------
    forecast_data = master_scaled.copy()
    lagged_features = []

    for column in FEATURE_COLUMNS:
        for lag in LAGS:
            feature_name = f"{column}_lag{lag}"
            forecast_data[feature_name] = (
                forecast_data[column].shift(lag)
            )
            lagged_features.append(feature_name)

    # Map next-month SCFI, matching the notebook.
    next_month_scfi = (
        master_scaled
        .set_index("Date")["SCFI"]
        .shift(-1)
    )

    forecast_data["Target_SCFI"] = (
        forecast_data["Date"].map(next_month_scfi)
    )

    forecast_data["Year"] = (
        forecast_data["Date"].dt.year
    )

    regression_data = (
        forecast_data
        .dropna(
            subset=lagged_features + ["Target_SCFI"]
        )
        .reset_index(drop=True)
    )

    if len(regression_data) < 36:
        raise ValueError(
            "At least 36 usable monthly observations are required "
            "after creating six lags."
        )

    # -------------------------
    # Walk-forward Linear Regression
    # -------------------------
    test_years = sorted(
        regression_data.loc[
            regression_data["Year"] >= 2019,
            "Year"
        ].unique()
    )

    wf_actual = []
    wf_predicted = []
    wf_dates = []

    for year in test_years:
        train = regression_data[
            regression_data["Year"] < year
        ].copy()

        test = regression_data[
            regression_data["Year"] == year
        ].copy()

        if test.empty or train.empty:
            continue

        X_train = train[lagged_features]
        X_test = test[lagged_features]
        y_train = train["Target_SCFI"]
        y_test = test["Target_SCFI"]

        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(
            X_train
        )
        X_test_scaled = fold_scaler.transform(
            X_test
        )

        fold_model = LinearRegression()
        fold_model.fit(X_train_scaled, y_train)

        predictions = fold_model.predict(
            X_test_scaled
        )

        wf_actual.extend(y_test.tolist())
        wf_predicted.extend(predictions.tolist())
        wf_dates.extend(test["Date"].tolist())

    if wf_actual:
        forecast_mae = mean_absolute_error(
            wf_actual,
            wf_predicted
        )

        forecast_rmse = np.sqrt(
            mean_squared_error(
                wf_actual,
                wf_predicted
            )
        )

        forecast_r2 = r2_score(
            wf_actual,
            wf_predicted
        )

        residual_std = np.std(
            np.array(wf_actual) - np.array(wf_predicted),
            ddof=1
        )
    else:
        forecast_mae = np.nan
        forecast_rmse = np.nan
        forecast_r2 = np.nan
        residual_std = np.nan

    walk_forward_predictions = pd.DataFrame({
        "Forecast_Origin": wf_dates,
        "Target_Date": [
            date + 1 for date in wf_dates
        ],
        "Actual_SCFI": wf_actual,
        "Predicted_SCFI": wf_predicted
    })

    if not walk_forward_predictions.empty:
        walk_forward_predictions["Residual"] = (
            walk_forward_predictions["Actual_SCFI"]
            - walk_forward_predictions["Predicted_SCFI"]
        )

    # -------------------------
    # Final one-month-ahead forecast
    # -------------------------
    X_final_train = regression_data[lagged_features]
    y_final_train = regression_data["Target_SCFI"]

    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(
        X_final_train
    )

    final_model = LinearRegression()
    final_model.fit(
        X_final_scaled,
        y_final_train
    )

    forecast_candidates = (
        forecast_data
        .dropna(subset=lagged_features)
        .copy()
    )

    latest_row = forecast_candidates.iloc[[-1]]
    forecast_origin = latest_row.iloc[0]["Date"]
    forecast_month = forecast_origin + 1

    next_forecast = float(
        final_model.predict(
            final_scaler.transform(
                latest_row[lagged_features]
            )
        )[0]
    )

    if np.isfinite(residual_std):
        lower_95 = (
            next_forecast - 1.96 * residual_std
        )
        upper_95 = (
            next_forecast + 1.96 * residual_std
        )
    else:
        lower_95 = np.nan
        upper_95 = np.nan

    forecast_class = classify_fragility(
        next_forecast
    )

    forecast_table = pd.DataFrame({
        "Forecast Origin": [str(forecast_origin)],
        "Forecasted Month": [str(forecast_month)],
        "Current SCFI": [
            float(master_scaled.iloc[-1]["SCFI"])
        ],
        "Forecasted SCFI": [next_forecast],
        "Approx. Lower 95%": [lower_95],
        "Approx. Upper 95%": [upper_95],
        "Forecasted Fragility": [forecast_class],
        "Selected Model": ["Linear Regression"],
        "Walk-forward MAE": [forecast_mae],
        "Walk-forward RMSE": [forecast_rmse],
        "Walk-forward R²": [forecast_r2]
    })

    merged_export = master.copy()
    merged_export["Date"] = (
        merged_export["Date"].astype(str)
    )

    standardised_export = master_scaled.copy()
    standardised_export["Date"] = (
        standardised_export["Date"].astype(str)
    )

    scfi_results = master_scaled[
        ["Date", "SCFI", "Fragility_Class"]
    ].copy()

    scfi_results["Date"] = (
        scfi_results["Date"].astype(str)
    )

    return {
        "data_quality": data_quality,
        "merged_data": merged_export,
        "standardised_data": standardised_export,
        "pca_table": pca_table,
        "scfi_results": scfi_results,
        "validation_table": validation_table,
        "forecast_table": forecast_table,
        "walk_forward_predictions": walk_forward_predictions,
        "master_scaled": master_scaled,
        "weights": dict(
            zip(VARIABLES, normalised_weights)
        ),
        "explained_variance": pca.explained_variance_ratio_,
        "lagged_features": lagged_features,
        "current_scfi": float(
            master_scaled.iloc[-1]["SCFI"]
        ),
        "current_class": master_scaled.iloc[-1][
            "Fragility_Class"
        ],
        "forecast_scfi": next_forecast,
        "forecast_class": forecast_class,
        "forecast_month": forecast_month
    }


# =========================================================
# SESSION STATE
# =========================================================

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None


# =========================================================
# SIDEBAR UPLOAD WORKFLOW
# =========================================================

with st.sidebar:
    st.header("1. Upload datasets")

    gscpi_file = st.file_uploader(
        "Upload GSCPI",
        type=["csv", "xls", "xlsx"],
        key="gscpi_upload"
    )

    inflation_file = st.file_uploader(
        "Upload inflation",
        type=["csv", "xls", "xlsx"],
        key="inflation_upload"
    )

    gpr_file = st.file_uploader(
        "Upload GPR",
        type=["csv", "xls", "xlsx"],
        key="gpr_upload"
    )

    oil_file = st.file_uploader(
        "Upload WTI oil data",
        type=["csv", "xls", "xlsx"],
        key="oil_upload"
    )

    if st.button(
        "Reset dashboard",
        use_container_width=True
    ):
        st.session_state.analysis_results = None
        st.rerun()


# =========================================================
# FILE CONFIGURATION
# =========================================================

files_uploaded = all([
    gscpi_file,
    inflation_file,
    gpr_file,
    oil_file
])

if not files_uploaded:
    st.info(
        "Upload all four datasets in the sidebar to begin. "
        "The application accepts CSV, XLS and XLSX files."
    )
    st.stop()

st.subheader("2. Configure uploaded files")

config_tabs = st.tabs([
    "GSCPI",
    "Inflation",
    "GPR",
    "Oil"
])


def upload_configuration(
    uploaded_file,
    prefix,
    suggested_date=None,
    suggested_value=None,
    default_skip=0
):
    sheet_name = None
    sheets = excel_sheet_names(uploaded_file)

    if sheets:
        sheet_index = (
            sheets.index("GSCPI Monthly Data")
            if "GSCPI Monthly Data" in sheets
            else 0
        )

        sheet_name = st.selectbox(
            "Worksheet",
            sheets,
            index=sheet_index,
            key=f"{prefix}_sheet"
        )

    raw = read_uploaded_file(
        uploaded_file,
        sheet_name=sheet_name
    )

    columns = list(raw.columns)

    date_index = (
        columns.index(suggested_date)
        if suggested_date in columns
        else 0
    )

    value_index = (
        columns.index(suggested_value)
        if suggested_value in columns
        else min(1, len(columns) - 1)
    )

    date_col = st.selectbox(
        "Date column",
        columns,
        index=date_index,
        key=f"{prefix}_date"
    )

    value_col = st.selectbox(
        "Value column",
        columns,
        index=value_index,
        key=f"{prefix}_value"
    )

    start_row = st.number_input(
        "Rows to skip before the data starts",
        min_value=0,
        max_value=max(0, len(raw) - 1),
        value=min(default_skip, max(0, len(raw) - 1)),
        step=1,
        key=f"{prefix}_skip"
    )

    st.caption(
        f"Detected {len(raw):,} rows and {len(columns)} columns."
    )
    st.dataframe(
        raw.head(8),
        use_container_width=True
    )

    return raw, date_col, value_col, start_row


with config_tabs[0]:
    gscpi_raw, gscpi_date, gscpi_value, gscpi_skip = (
        upload_configuration(
            gscpi_file,
            "gscpi",
            suggested_date="Date",
            suggested_value="GSCPI",
            default_skip=4
        )
    )

with config_tabs[1]:
    inflation_raw, inflation_date, inflation_value, inflation_skip = (
        upload_configuration(
            inflation_file,
            "inflation",
            suggested_date="TIME_PERIOD",
            suggested_value="OBS_VALUE",
            default_skip=0
        )
    )

with config_tabs[2]:
    gpr_raw, gpr_date, gpr_value, gpr_skip = (
        upload_configuration(
            gpr_file,
            "gpr",
            suggested_date="month",
            suggested_value="GPR",
            default_skip=0
        )
    )

with config_tabs[3]:
    oil_raw, oil_date, oil_value, oil_skip = (
        upload_configuration(
            oil_file,
            "oil",
            suggested_date="DATE",
            suggested_value="DCOILWTICO",
            default_skip=0
        )
    )

    oil_value_type = st.radio(
        "Uploaded oil value",
        [
            "Oil price — calculate monthly return",
            "Monthly oil return already calculated"
        ],
        key="oil_type"
    )


# =========================================================
# CALCULATE
# =========================================================

st.subheader("3. Validate and calculate the SCFI")

calculate_clicked = st.button(
    "Validate files and calculate SCFI",
    type="primary",
    use_container_width=True
)

if calculate_clicked:
    try:
        with st.spinner(
            "Cleaning datasets, calculating PCA weights, "
            "building the SCFI and forecasting next month..."
        ):
            gscpi = clean_two_column_data(
                gscpi_raw,
                gscpi_date,
                gscpi_value,
                "GSCPI",
                gscpi_skip
            )

            inflation = clean_two_column_data(
                inflation_raw,
                inflation_date,
                inflation_value,
                "Inflation_Rate",
                inflation_skip
            )

            gpr = clean_two_column_data(
                gpr_raw,
                gpr_date,
                gpr_value,
                "GPR",
                gpr_skip
            )

            oil_prepared_raw = oil_raw.iloc[
                int(oil_skip):
            ].copy()

            oil = clean_oil_data(
                oil_prepared_raw,
                oil_date,
                oil_value,
                oil_value_type
            )

            results = calculate_scfi_and_forecast(
                gscpi,
                inflation,
                gpr,
                oil
            )

            st.session_state.analysis_results = results

        st.success(
            "SCFI calculation and forecasting completed successfully."
        )

    except Exception as exc:
        st.session_state.analysis_results = None
        st.error("The analysis could not be completed.")
        st.exception(exc)


results = st.session_state.analysis_results

if results is None:
    st.stop()


# =========================================================
# OUTPUT DASHBOARD
# =========================================================

st.subheader("4. Analytical results")

top_1, top_2, top_3, top_4 = st.columns(4)

top_1.metric(
    "Current SCFI",
    f'{results["current_scfi"]:.4f}'
)

top_2.metric(
    "Current fragility",
    results["current_class"]
)

top_3.metric(
    f'Forecast SCFI ({results["forecast_month"]})',
    f'{results["forecast_scfi"]:.4f}'
)

top_4.metric(
    "Forecast fragility",
    results["forecast_class"]
)

if results["forecast_class"] == "Severe":
    st.error("The next-month forecast indicates Severe fragility.")
elif results["forecast_class"] == "High":
    st.warning("The next-month forecast indicates High fragility.")
else:
    st.success("The next-month forecast indicates Moderate fragility.")


result_tabs = st.tabs([
    "Overview",
    "Data Quality",
    "SCFI Development",
    "Validation",
    "Forecasting",
    "Downloads"
])


with result_tabs[0]:
    st.markdown("### Latest SCFI values")

    latest_rows = (
        results["standardised_data"]
        .tail(12)
        .copy()
    )

    st.dataframe(
        latest_rows[
            [
                "Date",
                "GSCPI",
                "Inflation_Rate",
                "GPR",
                "Oil_Return",
                "SCFI",
                "Fragility_Class"
            ]
        ],
        hide_index=True,
        use_container_width=True
    )

    class_counts = (
        results["standardised_data"][
            "Fragility_Class"
        ]
        .value_counts()
        .rename_axis("Fragility Class")
        .reset_index(name="Observations")
    )

    st.markdown("### Fragility distribution")
    st.dataframe(
        class_counts,
        hide_index=True,
        use_container_width=True
    )


with result_tabs[1]:
    st.markdown("### Uploaded dataset validation")

    st.dataframe(
        results["data_quality"],
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### Common merged observations")
    st.dataframe(
        results["merged_data"].head(20),
        hide_index=True,
        use_container_width=True
    )


with result_tabs[2]:
    st.markdown("### PCA-derived weights")

    pca_display = results["pca_table"].copy()

    for col in [
        "PC1 Loading",
        "Absolute Loading",
        "Normalised Weight",
        "Weight (%)"
    ]:
        pca_display[col] = pca_display[col].round(4)

    st.dataframe(
        pca_display,
        hide_index=True,
        use_container_width=True
    )

    formula_parts = [
        f'{weight:.3f}({variable}*)'
        for variable, weight
        in results["weights"].items()
    ]

    st.info(
        "SCFI = " + " + ".join(formula_parts)
    )

    history = results["master_scaled"].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["Date"].astype(str),
            y=history["SCFI"],
            mode="lines",
            name="SCFI"
        )
    )

    fig.add_hline(
        y=0.5,
        line_dash="dot",
        annotation_text="High threshold"
    )

    fig.add_hline(
        y=1.5,
        line_dash="dot",
        annotation_text="Severe threshold"
    )

    fig.update_layout(
        title="Supply Chain Fragility Index over Time",
        xaxis_title="Month",
        yaxis_title="SCFI",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with result_tabs[3]:
    st.markdown("### SCFI validation against GSCPI")

    validation_display = (
        results["validation_table"].copy()
    )
    validation_display["Value"] = (
        validation_display["Value"].round(4)
    )

    st.dataframe(
        validation_display,
        hide_index=True,
        use_container_width=True
    )

    history = results["master_scaled"].copy()

    comparison = go.Figure()

    comparison.add_trace(
        go.Scatter(
            x=history["Date"].astype(str),
            y=history["GSCPI"],
            mode="lines",
            name="GSCPI"
        )
    )

    comparison.add_trace(
        go.Scatter(
            x=history["Date"].astype(str),
            y=history["SCFI"],
            mode="lines",
            name="SCFI"
        )
    )

    comparison.update_layout(
        title="GSCPI and Proposed SCFI",
        xaxis_title="Month",
        yaxis_title="Standardised index value",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        comparison,
        use_container_width=True
    )

    difference = go.Figure()

    difference.add_trace(
        go.Scatter(
            x=history["Date"].astype(str),
            y=history["SCFI"] - history["GSCPI"],
            mode="lines",
            name="SCFI − GSCPI"
        )
    )

    difference.add_hline(y=0, line_dash="dash")

    difference.update_layout(
        title="Difference between SCFI and GSCPI",
        xaxis_title="Month",
        yaxis_title="Difference",
        height=420
    )

    st.plotly_chart(
        difference,
        use_container_width=True
    )


with result_tabs[4]:
    st.markdown("### Final one-month-ahead forecast")

    forecast_display = (
        results["forecast_table"].copy()
    )

    numeric_columns = [
        "Current SCFI",
        "Forecasted SCFI",
        "Approx. Lower 95%",
        "Approx. Upper 95%",
        "Walk-forward MAE",
        "Walk-forward RMSE",
        "Walk-forward R²"
    ]

    for col in numeric_columns:
        forecast_display[col] = (
            forecast_display[col].round(4)
        )

    st.dataframe(
        forecast_display,
        hide_index=True,
        use_container_width=True
    )

    history = results["master_scaled"].tail(36)

    forecast_chart = go.Figure()

    forecast_chart.add_trace(
        go.Scatter(
            x=history["Date"].astype(str),
            y=history["SCFI"],
            mode="lines+markers",
            name="Historical SCFI"
        )
    )

    forecast_origin = (
        results["forecast_table"]
        .iloc[0]["Forecast Origin"]
    )

    forecast_month = (
        results["forecast_table"]
        .iloc[0]["Forecasted Month"]
    )

    current_scfi = (
        results["forecast_table"]
        .iloc[0]["Current SCFI"]
    )

    predicted_scfi = (
        results["forecast_table"]
        .iloc[0]["Forecasted SCFI"]
    )

    forecast_chart.add_trace(
        go.Scatter(
            x=[forecast_origin, forecast_month],
            y=[current_scfi, predicted_scfi],
            mode="lines+markers",
            line=dict(dash="dash"),
            name="One-month forecast"
        )
    )

    forecast_chart.add_hline(
        y=0.5,
        line_dash="dot",
        annotation_text="High"
    )

    forecast_chart.add_hline(
        y=1.5,
        line_dash="dot",
        annotation_text="Severe"
    )

    forecast_chart.update_layout(
        title="Historical SCFI and One-Month-Ahead Forecast",
        xaxis_title="Month",
        yaxis_title="SCFI",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        forecast_chart,
        use_container_width=True
    )

    if not results["walk_forward_predictions"].empty:
        st.markdown(
            "### Walk-forward Linear Regression predictions"
        )

        st.dataframe(
            results["walk_forward_predictions"].tail(24),
            hide_index=True,
            use_container_width=True
        )


with result_tabs[5]:
    st.markdown("### Download analytical outputs")

    excel_bytes = dataframe_to_excel(results)
    html_bytes = create_html_report(results)

    d1, d2, d3 = st.columns(3)

    d1.download_button(
        "Download complete Excel report",
        data=excel_bytes,
        file_name="SCFI_Complete_Analysis.xlsx",
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
        use_container_width=True
    )

    d2.download_button(
        "Download SCFI results CSV",
        data=results["scfi_results"].to_csv(
            index=False
        ).encode("utf-8"),
        file_name="SCFI_Results.csv",
        mime="text/csv",
        use_container_width=True
    )

    d3.download_button(
        "Download analytical report",
        data=html_bytes,
        file_name="SCFI_Analytical_Report.html",
        mime="text/html",
        use_container_width=True
    )

    st.caption(
        "The Excel workbook includes data quality, merged data, "
        "standardised values, PCA weights, validation metrics, "
        "walk-forward forecasts and the final one-month forecast."
    )
