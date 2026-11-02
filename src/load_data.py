
import pandas as pd
from pathlib import Path

# Find the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Create the dataset path
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH, sep=";")

# Display first 5 rows
print("FIRST 5 STUDENTS:")
print(df.head())

# Display dataset information
print("\nDATASET INFORMATION:")
df.info()

# Display dataset shape
print("\nDATASET SHAPE:")
print(df.shape)

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Statistical summary
print("\nSTATISTICAL SUMMARY:")
print(df.describe())
