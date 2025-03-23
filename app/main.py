import streamlit as st  # Import Streamlit for UI
import pandas as pd  # Data handling
from app.config import DATA_PATH, MODEL_PATH  # Config paths
from data.data_utils import load_data, preprocess_input_data  # Data functions
from model.model_utils import load_model, predict  # Model functions
import datetime  # For date handling
from data.data_loader import get_df_train

# Load data and model
@st.cache_data  # Cache to improve performance
def get_data():
    return load_data(DATA_PATH)

@st.cache_resource  # Cache model loading to avoid reloading on every interaction
def get_model():
    return load_model(MODEL_PATH)

def main():
    st.title("Corporación Favorita Sales Forecasting Using XGBoost Model")
    df_train = get_df_train()

    # Load datasets and model
    df_stores, df_items, df_transactions, df_oil, df_holidays, df_train = get_data()
    model = get_model()

    # Store and item numbers provided
    store_numbers = [24, 28, 34, 51]  # Allowed store numbers
    item_numbers = [257847, 315176, 1463862, 1463814, 1047679, 1074327]  # Allowed item numbers

    # Sidebar inputs
    st.sidebar.header("Select Inputs")

    # Extract unique store numbers from df_train
    store_ids = df_train[df_train['store_nbr'].isin(store_numbers)]['store_nbr'].unique()

    # Store selection dropdown
    store_id = st.sidebar.selectbox("Select Store", store_ids)

    # Extract available items for the selected store and filter allowed items
    available_items = df_train[
        (df_train['store_nbr'] == store_id) & (df_train['item_nbr'].isin(item_numbers))
        ]['item_nbr'].unique()

    # Item selection dropdown
    item_id = st.sidebar.selectbox("Select Item", available_items)

    # Date range selection
    default_date = datetime.date(2014, 1, 1)  # Default start date
    min_date = datetime.date(2014, 1, 1)  # Start date of the dataset
    max_date = datetime.date(2014, 3, 31)  # End date of the dataset
    date = st.sidebar.date_input("Select Date", value=default_date, min_value=min_date, max_value=max_date)
    
    # Display historical sales trend
    historical_sales = df_train[
        (df_train["store_nbr"] == store_id) & 
        (df_train["item_nbr"] == item_id) & 
        (df_train["date"] >= "2013-01-16") &
        (df_train["date"] <= "2014-03-31")
    ]

    # Ensure date column is set as index and sorted
    historical_sales = historical_sales.set_index("date").sort_index()

    # Display the line chart for unit sales
    st.subheader("Historical Sales Data")
    if not historical_sales.empty:
        st.line_chart(historical_sales["unit_sales"])
    else:
        st.write("No sales data available for the selected store and item.") 

    # Predict button
    if st.sidebar.button("Get Forecast"):
        input_data = preprocess_input_data(store_id, item_id, date, df_train)
        prediction = predict(model, input_data)
        st.success(f"Predicted Sales for {date}: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()

