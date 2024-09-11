# Walmart Sales Prediction Using Machine Learning

## Introduction
This repository contains the code and analysis for predicting weekly Walmart sales using advanced machine learning techniques. Sales prediction is crucial in retail for managing inventory, optimizing supply chains, and improving customer satisfaction. Accurate sales forecasts allow Walmart to make data-driven decisions regarding product stocking, staffing, and promotions, which directly impacts profitability and operational efficiency.

## Objective
The objective of this analysis is to develop a predictive model for Walmart's weekly sales by exploring different machine learning algorithms. The goal is to compare the performance of **Random Forest Regression** and **ARIMA**, identifying the model that offers the best predictive accuracy. The final model will help Walmart better anticipate demand and optimize their operations.

## Dataset
The dataset used for this project includes weekly sales data for various Walmart stores across multiple departments. It also contains additional features such as markdowns, CPI, and unemployment data, which may influence sales.

### Data Dictionary

| Column Name        | Description                                        |
|--------------------|----------------------------------------------------|
| **Store**           | Store identifier                                  |
| **Dept**            | Department identifier                             |
| **Date**            | The week-end date of the week for which the data is recorded |
| **Weekly_Sales**    | Weekly sales for the given department in the store |
| **IsHoliday**       | Boolean flag indicating if the week is a special holiday week |
| **Temperature**     | Average temperature for that region               |
| **Fuel_Price**      | Cost of fuel in the region                        |
| **MarkDown1**       | First markdown data for promotional discounts     |
| **MarkDown2**       | Second markdown data                              |
| **MarkDown3**       | Third markdown data                               |
| **MarkDown4**       | Fourth markdown data                              |
| **MarkDown5**       | Fifth markdown data                               |
| **CPI**             | Consumer Price Index for that region              |
| **Unemployment**    | Unemployment rate for that region                 |
| **Type**            | Store type (A, B, C)                              |
| **Size**            | Size of the store                                 |
| **Holiday Indicators** | Various flags for holiday proximity like Thanksgiving and Christmas weeks |

## Approach
The approach followed includes a structured pipeline starting from data preprocessing to model evaluation. The key steps are:
1. **Data Preprocessing**: The data was cleaned, and missing values were handled appropriately. Time-based features were created, and the holiday impact was included. Markdown and economic features were analyzed for their contribution to sales.
2. **Exploratory Data Analysis**: Key trends were analyzed, including how sales varied across time, stores, and departments. Correlations between variables were also examined.
3. **Feature Engineering**: Additional features such as `Weekly_Sales_Median`, holiday flags, and markdown combinations were introduced to improve model performance.
4. **Modeling**: Both Random Forest and ARIMA models were applied. The dataset was split into training and test sets, and models were evaluated on multiple error metrics.
5. **Evaluation**: The models were compared based on their performance in predicting weekly sales, with particular focus on **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.

## Models Chosen
1. **Random Forest Regression**: 
   - Random Forest is an ensemble learning method known for its ability to handle high-dimensional datasets and interactions between variables. It was chosen due to its strong performance on tabular data with multiple features.
   - Hyperparameter tuning was performed using **GridSearchCV** to select the optimal number of trees and depth.

2. **ARIMA**: 
   - ARIMA (AutoRegressive Integrated Moving Average) is a classical time series forecasting model. It was chosen for its ability to model temporal dependencies in the data, although it does not inherently handle external variables like markdowns or holidays.
   - The ARIMA model was optimized using **auto_arima** for hyperparameter tuning.

## Results
The results from both models are summarized below:

| Model                | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
|----------------------|---------------------------|--------------------------------|
| **Random Forest**     | 2,615.61                  | 5,183.65                       |
| **ARIMA**             | 19,070.70                 | 30,332.46                      |

- **Random Forest** clearly outperformed **ARIMA**, achieving significantly lower error rates. The **absolute error plot** illustrated that Random Forest consistently provided more accurate predictions across the sales range, while ARIMA struggled with larger sales values.

## Conclusion
The analysis concluded that **Random Forest Regression** is the superior model for predicting Walmart's weekly sales. Its ability to capture interactions between features like markdowns, economic indicators, and holiday effects made it more suitable for the dataset compared to ARIMA, which only models temporal dependencies.

### Business Implications:
1. **Random Forest** can provide Walmart with more accurate sales forecasts, improving inventory management and reducing costs associated with overstock or stockouts.
2. The model's performance suggests that Walmart can leverage machine learning to optimize resource allocation and improve customer satisfaction by predicting high-demand periods more effectively.
3. Further enhancements could include incorporating additional external data, such as competitor activity or weather patterns, to improve predictions even further.
