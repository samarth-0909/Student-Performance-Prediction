
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "student-mat.csv", sep=";")

# Features and target
X = df.drop("G3", axis=1)
y = df["G3"]

# Identify columns
cat_cols = X.select_dtypes(include=["str", "object"]).columns
num_cols = X.select_dtypes(exclude=["str", "object"]).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

# Train and compare
for name, algorithm in models.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", algorithm)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel:", name)
    print("MAE:", round(mae, 3))
    print("R2 Score:", round(r2, 3))

# Train the final Random Forest model
best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

best_model.fit(X_train, y_train)

# Generate test predictions
y_pred = best_model.predict(X_test)

# Save model
MODEL_PATH = BASE_DIR / "models" / "student_performance_model.pkl"
joblib.dump(best_model, MODEL_PATH)

# Save actual and predicted grades
results = pd.DataFrame({
    "Actual Grade": y_test,
    "Predicted Grade": y_pred
})

results.to_csv(
    BASE_DIR / "reports" / "prediction_results.csv",
    index=False
)

print("\nModel and prediction results saved successfully!")

print("\nBest model saved successfully!")