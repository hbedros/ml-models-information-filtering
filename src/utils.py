
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

uri = "mongodb+srv://haigbedros:xqLlcSm2DP7VMGiF@cluster0.kfjsw.mongodb.net/?appName=Cluster0"

# Load .env file
load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv("MONGO_URI"))  # OR replace with your URI directly
db = client["my_database"]
collection = db["my_collection"]

# Pull documents from MongoDB and convert to DataFrame
cursor = collection.find().limit(5)  # you can remove .limit() for full data
df = pd.DataFrame(list(cursor))

# Preview
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
'''
# connect to mongodb
uri = "mongodb+srv://haigbedros:xqLlcSm2DP7VMGiF@cluster0.kfjsw.mongodb.net/?appName=Cluster0"


# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Load .env if using environment variable
load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv("MONGO_URI"))  # or paste your URI directly
db = client["my_database"]                    # Name your DB
collection = db["my_collection"]              # Name your collection

# Load data
df = pd.read_csv("/Users/haigbedros/Desktop/MSDS/Capstone/CODE/ml-models-information-filtering/Data/TFA_AlumniSurvey_2024.csv")

# Optionally clean or rename columns if needed
# df = df.rename(columns={"OldName": "new_name"})

# Convert DataFrame to list of dictionaries
data = df.to_dict("records")

# Insert into MongoDB
if data:
    collection.insert_many(data)
    print(f"✅ Successfully inserted {len(data)} documents.")
else:
    print("⚠️ No data found in CSV.")


'''




