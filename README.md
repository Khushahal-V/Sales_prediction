# Sales_prediction
Self project on walmart sales prediction dataset

Sales Forecasting Report – Walmart
Recruiting Challenge
Problem Statement
The challenge is to build a predictive model for forecasting department-level sales across
45 Walmart stores. The primary task is to estimate weekly sales for each department in
each store using historical sales data. A major difficulty lies in handling limited historical
observations for major events like Christmas or markdowns, which happen infrequently
but have significant impact on sales. The dataset includes holiday markdown events
known to affect demand patterns, making it important to incorporate these variables in
the forecasting model.



Data Sources Used
• train.csv: Historical weekly sales data by store and department.
• test.csv: Data for which predictions are required.
• stores.csv: Metadata about each store, such as type and size.
• features.csv: Additional information such as temperature, fuel prices, promotions
(markdowns), CPI, and holidays.



Key Steps Taken
• Loaded all datasets using pandas.
• Combined store and department into a unique identifier: Store Dept.
• Inspected null values, unique store types, and departments.
• Engineered new features to better capture department-store dynamics.
• Planned ARIMA modeling using statsmodels to capture trends and seasonality.


Parameters & Models
• ARIMA (AutoRegressive Integrated Moving Average) model.
• Applied time series techniques such as lag features and rolling averages to forecast
retail sales with seasonal patterns
• Evaluation metrics (assumed): MAE or RMSE.



Challenges Addressed
• Limited Event Data: Focused on high-impact but infrequent events like Christmas.
• Markdowns: Integrated markdown periods and their effects on sales.


Conclusion
A structured approach was followed to forecast department-level sales at Walmart us-
ing historical trends, markdown events, and time series modeling. By incorporating
retail-specific patterns and using ARIMA, the model is well-positioned to help strategic
decision-making in the retail environment.
