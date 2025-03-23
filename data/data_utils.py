# data/data_utils.py
import pandas as pd
import os
import gdown
from app.config import DATA_PATH, GOOGLE_DRIVE_LINKS

def download_file(file_path, url):
    """Downloads a file from Google Drive if it doesn't exist locally."""
    if not os.path.exists(file_path):
        gdown.download(url, file_path, quiet=False)
    else:
        print(f"{file_path} already exists.")

def load_data(data_path=DATA_PATH):
    """Downloads necessary data from Google Drive and loads CSV files into DataFrames."""
    
    # Define the paths for all the required data files
    files = {
        "stores": f"{data_path}stores.csv",  # Path for stores data
        "items": f"{data_path}items.csv",  # Path for items data
        "transactions": f"{data_path}transactions.csv",  # Path for transactions data
        "oil": f"{data_path}oil.csv",  # Path for oil prices data
        "holidays_events": f"{data_path}holidays_events.csv",  # Path for holidays and events data
        "train": f"{data_path}train.csv"  # Path for training data
    }

    # Download the files if they don't already exist locally
    for key, file_path in files.items():
        download_file(file_path, GOOGLE_DRIVE_LINKS[key])

    # Load each downloaded CSV file into a pandas DataFrame
    df_stores = pd.read_csv(files["stores"])  # Stores data
    df_items = pd.read_csv(files["items"])  # Items data
    df_transactions = pd.read_csv(files["transactions"])  # Transactions data
    df_oil = pd.read_csv(files["oil"])  # Oil prices data
    df_holidays = pd.read_csv(files["holidays_events"])  # Holidays and events data

    # Filter store numbers for the 'Guayas' state
    # Extract the unique store numbers from the 'Guayas' state in the stores dataframe
    store_ids = df_stores[df_stores['state'] == 'Guayas']['store_nbr'].unique()

    # Define the item families we want to filter: 'GROCERY I', 'BEVERAGES', 'CLEANING'
    item_families = ['GROCERY I', 'BEVERAGES', 'CLEANING']

    # Get item numbers that belong to the specified item families
    items_ids = df_items[df_items['family'].isin(item_families)]

    # Select data before April'14
    max_date = '2014-04-01'

    # Chunk size
    chunk_size = 10 ** 6

    # Create an empty list to store filtered chunks of data
    filtered_chunks = []

    # Loop through each chunk of data (for large dataset processing)
    for chunk in pd.read_csv(files["train"], chunksize=chunk_size,parse_dates=['date'],low_memory=False):
        # Filter the chunk based on store numbers, item numbers
        # Conditions:
        # - Store numbers should be in 'Guayas' state
        # - Item numbers should belong to the selected item families
        chunk_filtered = chunk[(chunk['store_nbr'].isin(store_ids))]
        chunk_filtered = chunk_filtered[(chunk_filtered['date'] < max_date)]
        chunk_filtered = chunk_filtered.merge(items_ids, on="item_nbr", how="inner")

        # Append the filtered chunk to the list of filtered chunks
        filtered_chunks.append(chunk_filtered)

        # Delete the chunk to free up memory (important for large datasets)
        del chunk

    # Combine all filtered chunks into a single DataFrame
    df_train = pd.concat(filtered_chunks, ignore_index=True)

    # Clean up the memory by deleting the list of filtered chunks
    del filtered_chunks
    # Return all the loaded DataFrames
    return df_stores, df_items, df_transactions, df_oil, df_holidays, df_train

def preprocess_input_data(store_id, item_id, date, df_train):
    print(f"🔍 Processing Store: {store_id}, Item: {item_id}, Date: {date}")  # Debugging

    """Preprocesses input data into a format suitable for model prediction."""
    # Convert the 'date' column to datetime format for easy manipulation
    df_train['date'] = pd.to_datetime(df_train['date'])

    # Get the minimum and maximum dates in the dataset to create a full date range
    min_date = df_train['date'].min()
    max_date = df_train['date'].max()
    print("Before filtering", min_date.date(), max_date.date())
    # Get the minimum and maximum dates in the dataset to create a full date range

    # Generate a full date range from min_date to max_date (daily frequency)
    full_date_range = pd.DataFrame({'date': pd.date_range(min_date, max_date, freq='D')})

    # Create a DataFrame with all (store, item, date) combinations by merging store-item pairs with full date range
    store_item_combinations = df_train[['store_nbr', 'item_nbr']].drop_duplicates()
    all_combinations = store_item_combinations.merge(full_date_range, how='cross')

    # Merge the full combinations with the original df_train to fill in missing sales for specific dates
    df_filled = all_combinations.merge(df_train, on=['store_nbr', 'item_nbr', 'date'], how='left')

    # Fill missing sales values with 0 (for days with no sales)
    df_filled['unit_sales'] = df_filled['unit_sales'].fillna(0)
    # Ensure no negative unit sales
    df_filled["unit_sales"] = df_filled["unit_sales"].apply(lambda x: max(x, 0))

    df_filled = df_filled.drop(columns=['onpromotion'])  # drop the onpromotion column
    df_filled = df_filled.drop(columns=['id'])  # drop the id column
    df_filled = df_filled.drop(columns=['perishable'])
    df_filled = df_filled.drop(columns=['family'])
    df_filled = df_filled.drop(columns=['class'])

    # New time-based features
    df_filled["year"] = df_filled["date"].dt.year
    df_filled["month"] = df_filled["date"].dt.month
    df_filled["day"] = df_filled["date"].dt.day
    df_filled["day_of_week"] = df_filled["date"].dt.dayofweek

    # Create lag features (previous sales)
    df_filled["lag_1"] = df_filled.groupby(["store_nbr", "item_nbr"])["unit_sales"].shift(1)
    df_filled["lag_7"] = df_filled.groupby(["store_nbr", "item_nbr"])["unit_sales"].shift(7)
    df_filled["lag_14"] = df_filled.groupby(["store_nbr", "item_nbr"])["unit_sales"].shift(14)

    # Rolling average of unit sales
    df_filled["rolling_avg_7"] = (df_filled.groupby(["store_nbr", "item_nbr"])["unit_sales"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()))

    df_filled["rolling_stdv_7"] = (df_filled.groupby(["store_nbr", "item_nbr"])["unit_sales"].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()))

    # Drop rows with NaN values after creating lag features
    df_filled = df_filled.dropna().reset_index(drop=True)
    # ✅ **NOW filter for the requested store, item, and date**
    df_filtered = df_filled[
        (df_filled["store_nbr"] == store_id) &
        (df_filled["item_nbr"] == item_id) &
        (df_filled["date"] == pd.to_datetime(date))
    ]

    if df_filtered.empty:
        print("⚠️ Warning: No matching data found! Check store/item selection.")
        return None  # Return None if no data matches

    # Drop unnecessary columns before passing to the model
    df_filtered = df_filtered.drop(columns=["unit_sales", "date"], errors="ignore")

    print("✅ Filtered Data Sample for Prediction:")
    print(df_filtered.head())  # Debugging

    return df_filtered