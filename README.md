# **Time Series Forecast Predicting Demand for Corporacion Favorita Guayas Retail Store**

This project aims to develop a **Machine Learning**-based sales forecasting model for different products across stores in the Guayas region. The predictions are visualized using a Streamlit web application, which allows users to select a store, item, and date to obtain sales forecasts.

---

## 🚀 **Project Overview**

- Conducted exploratory data analysis (EDA) to understand sales trends.
- Preprocessed data, including handling missing values and feature engineering.
- Developed a forecasting model to predict sales for January–March 2014.
- Built a Streamlit application for demand planners to interact with forecasts.

---

## 🔍 **Data Preprocessing**

- Filled missing dates with zero sales to ensure a continuous time series.
- Created lag features for trend analysis.
- Dropped rows with NaN values post-feature engineering.
- Merged data at the end of preprocessing to ensure consistency.

---

## 🤖 **Model Choice & Justification**

Different models were tested and selected **XGBoost** for its best result and ability to capture complex patterns and interactions within the data.

---
## **Model Performance**

- **RMSE (Root Mean Squared Error):** 5.98  
  This metric indicates that, on average, the model's predictions deviate from the actual sales by 5.98 units. A lower RMSE reflects better model accuracy.

- **MAE (Mean Absolute Error):** 2.24
  This metric measures the average magnitude of the errors in the predictions. Smaller values indicate better model performance.
---

## 🔧 **Hyperparameter Tuning and Model Evaluation**

Performed **GridSearchCV** for hyperparameter tuning to optimize the performance of the XGBoost model. The tuning process involved evaluating different combinations of hyperparameters and selecting the best model based on **Root Mean Squared Error (RMSE)**.

### **Key Steps in Hyperparameter Tuning:**
- **Hyperparameter Grid Search:**  
  A parameter grid was defined to test various values for `n_estimators`, `learning_rate`, and `max_depth`.
  
- **TimeSeriesSplit for Cross-Validation:**  
  Used to maintain the chronological order of the data, preventing data leakage.

- **GridSearchCV:**  
  An exhaustive search was performed over the grid to identify the best hyperparameters.

### **Best Hyperparameters:**
- **Learning Rate:** 0.1
- **Max Depth:** 5
- **Number of Estimators:** 50

### **Evaluation Metrics:**
- **Root Mean Squared Error (RMSE):** 5.76  
  This indicates the average deviation between predicted and actual sales values, with lower values reflecting better performance.

---

## 🌍 **Streamlit Web Application**

My Streamlit app allows users to:

- Select a store and item to forecast.
- Pick a date within the prediction range.
- Generate sales predictions interactively.

---

## **App Screenshots**
- **Store 28, Item 257847 - Forecast for 2014-02-01**
  - **Predicted Sales for 2014-02-01:** 57.68
  - **Actual Sales for 2014-02-01:** 63.00
![Alt text](Screenshot_Streamlit/Screenshot%202025-03-23%20at%2013.40.37.png)
[Watch the video](https://github.com/JyotiTidke26/Time-Series-Forecast-Predicting-Demand-for-Corporaci-n-Favorita-Guayas-Retail-Store/raw/main/Screenshot_Streamlit/Screen%20Recording%20-%20Mar%2023,%202025.mp4)

---

## **Dataset Download Instructions**

To download the dataset for the "Favorita Grocery Sales Forecasting" , follow these steps:

1. Go to the [Kaggle dataset page](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data).
2. Scroll down the page and click on the **“Download All”** button under the **Data** tab.
3. Once the files are downloaded, you can use them for your project.

Make sure to log in to your Kaggle account before downloading the files.


