import os
import sys
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import kagglehub
from kagglehub import KaggleDatasetAdapter
from utils.preprocessing import clean_text


# === Add src Directory to Python Path ===
src_path = os.path.abspath("/Users/haigbedros/Desktop/MSDS/Capstone/CODE/ml-models-information-filtering/src")
if src_path not in sys.path:
    sys.path.append(src_path)

# === Load Environment Variables ===
load_dotenv()

# === MongoDB Connection Setup ===
uri = "mongodb+srv://haigbedros:xqLlcSm2DP7VMGiF@cluster0.kfjsw.mongodb.net/?appName=Cluster0"
mongo_client = MongoClient(uri)

# === MongoDB Databases and Collections ===
tfa_db = mongo_client["tfa"]
tfa_collection = tfa_db["alum_survey_22"]

elmr_db = mongo_client["elmr"]
elmr_collection = elmr_db["elmr_book"]

elmr_compl_db = mongo_client["elmr"]
elmr_compl_collection = elmr_compl_db["elmr_book"]

marr_db = mongo_client["marr"]
marr_collection = marr_db["marr_book"]

marr_compl_db = mongo_client["marr_compl"]
marr_compl_collection = marr_compl_db["marr_compl_book"]

# === MongoDB Data Retrieval Functions with Preprocessing ===
def get_mongo_text(collection, concatenate=False, preprocess=False, **preprocess_kwargs):
    cursor = collection.find({}, {"text": 1, "_id": 0})
    df = pd.DataFrame(list(cursor))
    if preprocess:
        df['text'] = df['text'].dropna().apply(lambda x: clean_text(x, **preprocess_kwargs))
    if concatenate:
        return " ".join(df['text'].dropna().tolist())
    return df

def get_elmr_text(concatenate=False, preprocess=False, **preprocess_kwargs):
    return get_mongo_text(elmr_collection, concatenate, preprocess, **preprocess_kwargs)

def get_marr_text(concatenate=False, preprocess=False, **preprocess_kwargs):
    return get_mongo_text(marr_collection, concatenate, preprocess, **preprocess_kwargs)

def load_elmr_compl_df():
    elmr_compl_cursor = elmr_compl_collection.find()
    return pd.DataFrame(list(elmr_compl_cursor))
    

def load_marr_compl_df():
    marr_compl_cursor = marr_compl_collection.find()
    return pd.DataFrame(list(marr_compl_cursor))

# === Kaggle Dataset Loader ===
def load_kaggle_csv(dataset_id, file_path):
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dataset_id,
        file_path,
        pandas_kwargs={
            "on_bad_lines": "skip",
            "low_memory": False
        }
    )

# === Utility Function for Preprocessing DataFrame Columns ===
def preprocess_dataframe_column(df, column_name, **preprocess_kwargs):
    if column_name in df.columns:
        df[column_name] = df[column_name].dropna().apply(lambda x: clean_text(str(x), **preprocess_kwargs))
    return df

# === Load Datasets ===
# MongoDB Dataset (sampled small for testing)
def load_tfa_df(limit=5):
    tfa_cursor = tfa_collection.find().limit(limit)
    return pd.DataFrame(list(tfa_cursor))

def load_elmr_df(limit=5):
    elmr_cursor = elmr_collection.find().limit(limit)
    return pd.DataFrame(list(elmr_cursor))

def load_marr_df():
    marr_cursor = marr_collection.find()
    return pd.DataFrame(list(marr_cursor))

# Kaggle Datasets Loader
def load_all_kaggle_data():
    fake_df = load_kaggle_csv("emineyetm/fake-news-detection-datasets", "News _dataset/Fake.csv")
    true_df = load_kaggle_csv("emineyetm/fake-news-detection-datasets", "News _dataset/True.csv")
    nyt_df = load_kaggle_csv("kevinbnisch/ny-times-vectordb-for-topic-extraction", "2023-11-07_times_sum_1.csv")
    
    # Preprocess
    fake_df = preprocess_dataframe_column(fake_df, "text")
    true_df = preprocess_dataframe_column(true_df, "text")
    nyt_df = preprocess_dataframe_column(nyt_df, "text")
    
    return fake_df, true_df, nyt_df