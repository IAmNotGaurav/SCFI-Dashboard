# Upload-Driven SCFI Streamlit Dashboard

This Streamlit application reproduces the core analytical workflow from the final SCFI notebook.

## Main capabilities

- Four separate upload controls for GSCPI, inflation, GPR and oil datasets
- CSV, XLS and XLSX support
- Worksheet and column mapping
- Data preview and validation
- Monthly alignment and oil-return calculation
- PCA-derived SCFI weights
- Moderate, High and Severe fragility classification
- Validation against GSCPI using correlation, RMSE and MAE
- Six-month lag engineering
- Expanding walk-forward Linear Regression evaluation
- Genuine one-month-ahead SCFI forecast
- Downloadable Excel, CSV and HTML analytical reports

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected original notebook mappings

### GSCPI
- Worksheet: `GSCPI Monthly Data`
- Date: `Date`
- Value: `GSCPI`
- Rows skipped: 4

### OECD inflation
- Date: `TIME_PERIOD`
- Value: `OBS_VALUE`

### GPR
- Date: `month`
- Value: `GPR`

### WTI oil
- Date: normally `DATE`
- Value: normally `DCOILWTICO`
- Select `Oil price — calculate monthly return`

The mappings are configurable in the application, allowing similarly structured replacement datasets to be used.
