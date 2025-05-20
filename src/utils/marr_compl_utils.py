from pymongo import MongoClient
import pandas as pd

# === MongoDB Connection ===
uri = "mongodb+srv://haigbedros:xqLlcSm2DP7VMGiF@cluster0.kfjsw.mongodb.net/?appName=Cluster0"
mongo_client = MongoClient(uri)

# Access the marr_compl_db and marr_compl_collection
marr_compl_db = mongo_client["marr_compl"]
marr_compl_collection = marr_compl_db["marr_compl_book"]

# Load all documents from the collection into a DataFrame
marr_compl_cursor = marr_compl_collection.find()
marr_compl_df = pd.DataFrame(list(marr_compl_cursor)) 