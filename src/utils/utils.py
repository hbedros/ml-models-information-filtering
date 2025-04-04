import os
import sys
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import kagglehub
from kagglehub import KaggleDatasetAdapter
from utils.preprocessing import clean_text  # Import the clean_text function

# === Add src Directory to Python Path ===
src_path = os.path.abspath("/Users/haigbedros/Desktop/MSDS/Capstone/CODE/ml-models-information-filtering/src")
if src_path not in sys.path:
    sys.path.append(src_path)

# === Load Environment Variables ===
load_dotenv()

# === MongoDB Connection Setup ===
uri = "mongodb+srv://haigbedros:xqLlcSm2DP7VMGiF@cluster0.kfjsw.mongodb.net/?appName=Cluster0"
mongo_client = MongoClient(uri)

# MongoDB Databases and Collections
tfa_db = mongo_client["tfa"]
tfa_collection = tfa_db["alum_survey_22"]

elmr_db = mongo_client["elmr"]
elmr_collection = elmr_db["elmr_book"]

marr_db = mongo_client["marr"]
marr_collection = marr_db["marr_book"]

# === MongoDB Data Retrieval Functions with Preprocessing ===
def get_mongo_text(collection, concatenate=False, preprocess=False, **preprocess_kwargs):
    """
    Retrieve the 'text' field from a MongoDB collection and optionally preprocess it.

    Args:
        collection: MongoDB collection object.
        concatenate (bool): If True, concatenate all rows into a single string.
        preprocess (bool): If True, preprocess the text data.
        preprocess_kwargs: Additional arguments for the clean_text function.

    Returns:
        str or pd.DataFrame: Concatenated text (if concatenate=True) or a DataFrame with the 'text' column.
    """
    cursor = collection.find({}, {"text": 1, "_id": 0})
    df = pd.DataFrame(list(cursor))
    
    if preprocess:
        df['text'] = df['text'].dropna().apply(lambda x: clean_text(x, **preprocess_kwargs))
    
    if concatenate:
        return " ".join(df['text'].dropna().tolist())  # Drop NaN values and join all rows
    
    return df

# Update other functions to support preprocessing
def get_elmr_text(concatenate=False, preprocess=False, **preprocess_kwargs):
    return get_mongo_text(elmr_collection, concatenate, preprocess, **preprocess_kwargs)

def get_marr_text(concatenate=False, preprocess=False, **preprocess_kwargs):
    return get_mongo_text(marr_collection, concatenate, preprocess, **preprocess_kwargs)

# === Kaggle Dataset Loader ===
def load_kaggle_csv(dataset_id, file_path):
    """
    Load a CSV file from a Kaggle dataset using KaggleHub.

    Args:
        dataset_id (str): Kaggle dataset identifier.
        file_path (str): Path to the CSV file within the dataset.

    Returns:
        pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dataset_id,
        file_path,
        pandas_kwargs={
            "on_bad_lines": "skip",  # This replaces error_bad_lines=False in recent pandas
            "low_memory": False      # Optional: helps with mixed types
        }
    )

# === Utility Function for Preprocessing DataFrame Columns ===
def preprocess_dataframe_column(df, column_name, **preprocess_kwargs):
    """
    Preprocess a specific column in a DataFrame using the clean_text function.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        column_name (str): The name of the column to preprocess.
        preprocess_kwargs: Additional arguments for the clean_text function.

    Returns:
        pd.DataFrame: The DataFrame with the specified column preprocessed.
    """
    if column_name in df.columns:
        df[column_name] = df[column_name].dropna().apply(lambda x: clean_text(str(x), **preprocess_kwargs))
    return df

# === Load Datasets ===
# MongoDB Dataset
tfa_cursor = tfa_collection.find().limit(5)  # Limit to 5 for testing; remove .limit() for full data
tfa_df = pd.DataFrame(list(tfa_cursor))

marr_cursor = marr_collection.find().limit(5)  # Limit to 5 for testing; remove .limit() for full data
marr_df = pd.DataFrame(list(marr_cursor))

elmr_cursor = elmr_collection.find().limit(5)  # Limit to 5 for testing; remove .limit() for full data
elmr_df = pd.DataFrame(list(elmr_cursor))

# Preprocess MongoDB Datasets
# tfa_df = preprocess_dataframe_column(tfa_df, "text")
marr_df = preprocess_dataframe_column(marr_df, "text")
elmr_df = preprocess_dataframe_column(elmr_df, "text")

# Kaggle Datasets
fake_df = load_kaggle_csv("emineyetm/fake-news-detection-datasets", "News _dataset/Fake.csv")
true_df = load_kaggle_csv("emineyetm/fake-news-detection-datasets", "News _dataset/True.csv")
nyt_df = load_kaggle_csv("kevinbnisch/ny-times-vectordb-for-topic-extraction", "2023-11-07_times_sum_1.csv")

# Preprocess Kaggle Datasets
fake_df = preprocess_dataframe_column(fake_df, "text")
true_df = preprocess_dataframe_column(true_df, "text")
nyt_df = preprocess_dataframe_column(nyt_df, "text")