
import pandas as pd
import joblib
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load saved model
model = joblib.load(
    BASE_DIR / "models" / "student_performance_model.pkl"
)

# Load dataset
df = pd.read_csv(
    BASE_DIR / "data" / "student-mat.csv",
    sep=";"
)

# Select one student (first record)
student = df.drop("G3", axis=1).iloc[[0]]

# Predict final grade
prediction = model.predict(student)

print("Student Details:")
print(student)

print("\nPredicted Final Grade:", round(prediction[0], 2))
