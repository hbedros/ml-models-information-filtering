import sys
import os

# Define the main src path
src_path = os.path.abspath("/Users/haigbedros/Desktop/MSDS/Capstone/CODE/ml-models-information-filtering/src")

# Add the src directory to the Python path if not already present
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Print the updated Python path for verification
print("Updated Python Path:", sys.path)