import os

# Directory paths for data and model files
DATA_PATH = os.path.join(os.getcwd(), "data/")
MODEL_PATH = 'model/'  # Path to the directory containing the model files

# Google Drive file IDs for each dataset
your_file_id_for_stores_csv = '143dwSOX_n4wg_yqc30Vp7wtULpquPiWJ'  # ID for stores data CSV
your_file_id_for_items_csv = '1jhCeLojyAqE9bXE6WBb4uLCj-nOfLWfR'  # ID for items data CSV
your_file_id_for_transactions_csv = '17GZgm87IJ7TgBYKsSEl6FAWZh2dJgDrS'  # ID for transactions data CSV
your_file_id_for_oil_csv = '1ZuQUOz83wiOvSHRVkswJeTHGnYItZ6H0'  # ID for oil prices data CSV
your_file_id_for_holidays_csv = '1v5VkBc4s-E_BktDAi8mpS00cmm0F9WEz'  # ID for holidays data CSV
your_file_id_for_train_csv = '1JnNKp2p2QgPfy5sB5tbuKc2K84gRHqe8'  # ID for training data CSV

# Google Drive links for each dataset
GOOGLE_DRIVE_LINKS = {
    "stores": f"https://drive.google.com/uc?id={your_file_id_for_stores_csv}",  # Link for stores data
    "items": f"https://drive.google.com/uc?id={your_file_id_for_items_csv}",  # Link for items data
    "transactions": f"https://drive.google.com/uc?id={your_file_id_for_transactions_csv}",  # Link for transactions data
    "oil": f"https://drive.google.com/uc?id={your_file_id_for_oil_csv}",  # Link for oil prices data
    "holidays_events": f"https://drive.google.com/uc?id={your_file_id_for_holidays_csv}",  # Link for holidays data
    "train": f"https://drive.google.com/uc?id={your_file_id_for_train_csv}" # Link for training data
}

# Google Drive link for the model
your_file_id_for_xgboost_model_xgb = '1Nvl9_DFdAwOVYXJehdS8SyY1THI4ZrlF'  # ID for the XGBoost model file

# Google Drive link for the model file
GOOGLE_DRIVE_LINKS_MODELS = {
    "xgboost_model": f"https://drive.google.com/uc?id={your_file_id_for_xgboost_model_xgb}"  # Link for the XGBoost model
}
