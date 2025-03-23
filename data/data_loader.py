import pandas as pd
from app.config import DATA_PATH, GOOGLE_DRIVE_LINKS
from data.data_utils import load_data  # Import existing load_data function

def get_df_train():
    df_stores, df_items, df_transactions, df_oil, df_holidays, df_train = load_data(DATA_PATH)
    return df_train