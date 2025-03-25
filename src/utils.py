import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import kagglehub
from kagglehub import KaggleDatasetAdapter

# === Load TFA's Datasets from MongoDB ===
uri = "mongodb+srv://haigbedros:xqLlcSm2DP7VMGiF@cluster0.kfjsw.mongodb.net/?appName=Cluster0"

# Load environment variables
load_dotenv()

# Connect to MongoDB
tfa_client = MongoClient(os.getenv("MONGO_URI"))
tfa_db = tfa_client["tfa"]
tfa_collection = tfa_db["alum_survey_22"]

# Pull TFA documents from MongoDB and convert to DataFrame
tfa_cursor = tfa_collection.find().limit(5)  # you can remove .limit() for full data
tfa_df = pd.DataFrame(list(tfa_cursor))

# === Load Kaggle Datasets ===
def load_kaggle_csv(dataset_id, file_path):
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dataset_id,
        file_path,
        pandas_kwargs={
            "on_bad_lines": "skip",  # This replaces error_bad_lines=False in recent pandas
            "low_memory": False      # Optional: helps with mixed types
        }
    )


# Fake and True news datasets
fake_df = load_kaggle_csv("emineyetm/fake-news-detection-datasets", "News _dataset/Fake.csv")
true_df = load_kaggle_csv("emineyetm/fake-news-detection-datasets", "News _dataset/True.csv")

# NYT corpus
nyt_df = load_kaggle_csv("kevinbnisch/ny-times-vectordb-for-topic-extraction", "2023-11-07_times_sum_1.csv")

# Preview sample data
print("TFA Sample:", tfa_df.head(2))
print("Fake News Sample:", fake_df.head(2))
print("True News Sample:", true_df.head(2))
print("NYT Sample:", nyt_df.head(2))