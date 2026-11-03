import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Locate project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
df = pd.read_csv(BASE_DIR / "data" / "student-mat.csv", sep=";")

# Separate features and target
X = df.drop("G3", axis=1)
y = df["G3"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["str", "object"]).columns
numerical_cols = X.select_dtypes(exclude=["str", "object"]).columns

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Display results
print("Training Features:", X_train_processed.shape)
print("Testing Features:", X_test_processed.shape)
print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

print("\nPreprocessing completed successfully!")