# NOTES

## 2026-03-29T12:18:33Z
### Facts
- The last 10 records from SALES_REFUNDS are: 2024-10-19: 2771.56, 2024-10-20: 2300.83, 2024-10-21: 2260.7, 2024-10-22: 0, 2024-10-23: 8436.4, 2024-10-24: 0, 2024-10-25: 168.3, 2024-10-26: 503.582, 2024-10-27: 6201.68, 2024-10-28: 0

## 2026-03-29T12:32:44Z
### Facts
- Key column: BOOKING_DATE
- Target column: REFUNDS
- Data length: 150 days (2024-05-01 to 2024-09-28)
- 14% zero values (not highly intermittent)
- Series is stationary (no strong trend)
- No trend detected
- Additive seasonality detected with a period of 28 days
### Preferences
- Automatic Time Series Forecast for automation and tuning
- Additive Model Forecast for transparency and explicit seasonality

## 2026-03-29T12:33:34Z
### Facts
- Automatic Time Series Forecast model was used
- SALES_REFUNDS_TRAIN table was the training data
- Model saved as my_hana_ai_model

## 2026-03-29T12:34:26Z
### Facts
- Prediction results will be available in PREDICT_RESULT_SALES_REFUNDS_PREDICT_my_hana_ai_model_2.

## 2026-03-29T12:34:41Z
### Facts
- Predicted refunds for dates from 2024-09-29 to 2024-10-28 are available from the model my_hana_ai_model.

## 2026-03-29T12:35:05Z
### Facts
- There is a predicted results table.
- There is an actual table named SALES_REFUNDS.

## 2026-03-29T12:35:11Z
### Facts
- Project name is my_project
- Output path is cap

