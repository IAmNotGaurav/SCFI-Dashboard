# SCFI Dashboard

A portfolio-ready Streamlit decision-support application based on the final
Supply Chain Fragility Index dissertation notebook.

## Included

- Upload/configuration for GSCPI, OECD inflation, GPR and WTI oil files
- PCA-weighted SCFI calculation
- Data-quality checks
- Current and forecast fragility KPIs
- Major disruption event explorer
- PCA contribution charts
- GSCPI vs SCFI validation
- Walk-forward comparison of:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- Forecast model selector
- One-month-ahead SCFI forecast
- Approximate 95% forecast interval
- What-if scenario simulator
- Automated management briefing
- Downloadable:
  - Excel analytical workbook
  - PDF executive report
  - SCFI history CSV

- `README.md`

Your Streamlit app should point to `app.py`.

When committed to GitHub, Streamlit Community Cloud normally rebuilds the app
automatically. If it does not, choose **Manage app → Reboot app**.
