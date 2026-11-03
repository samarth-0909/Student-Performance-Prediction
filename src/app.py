
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "student_performance_model.pkl"
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"

# Page configuration
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=";")

model = load_model()
df = load_data()

# Sidebar
st.sidebar.title("🎓 Student Analytics")
page = st.sidebar.radio(
    "Navigation",
    ["Grade Prediction", "Dataset Overview", "Model Performance"]
)

# Main title
st.title("🎓 Student Performance Prediction System")
st.caption("Machine Learning Based Academic Performance Analysis")

if page == "Grade Prediction":

    st.header("Student Grade Prediction")
    st.write("Enter student details to predict their final grade.")

    student = df.drop("G3", axis=1).iloc[0].copy()
    inputs = {}

    with st.form("prediction_form"):

        st.subheader("👤 Personal Information")

        col1, col2, col3 = st.columns(3)

        personal_cols = [
            "school", "sex", "age", "address",
            "famsize", "Pstatus"
        ]

        for i, column in enumerate(personal_cols):
            with [col1, col2, col3][i % 3]:
                if df[column].dtype == "object" or df[column].dtype == "str":
                    options = sorted(df[column].dropna().unique().tolist())
                    inputs[column] = st.selectbox(column, options,
                        index=options.index(student[column]))
                else:
                    inputs[column] = st.number_input(
                        column,
                        min_value=int(df[column].min()),
                        max_value=int(df[column].max()),
                        value=int(student[column])
                    )

        st.divider()
        st.subheader("📚 Academic Information")

        academic_cols = [
            "Medu", "Fedu", "Mjob", "Fjob",
            "reason", "guardian", "traveltime",
            "studytime", "failures", "schoolsup",
            "famsup", "paid", "activities",
            "nursery", "higher", "internet",
            "romantic", "G1", "G2"
        ]

        col1, col2, col3 = st.columns(3)

        for i, column in enumerate(academic_cols):
            with [col1, col2, col3][i % 3]:
                if df[column].dtype == "object" or df[column].dtype == "str":
                    options = sorted(df[column].dropna().unique().tolist())
                    inputs[column] = st.selectbox(column, options,
                        index=options.index(student[column]))
                else:
                    inputs[column] = st.number_input(
                        column,
                        min_value=int(df[column].min()),
                        max_value=int(df[column].max()),
                        value=int(student[column])
                    )

        st.divider()
        st.subheader("🏠 Lifestyle Information")

        lifestyle_cols = [
            "famrel", "freetime", "goout",
            "Dalc", "Walc", "health", "absences"
        ]

        col1, col2, col3 = st.columns(3)

        for i, column in enumerate(lifestyle_cols):
            with [col1, col2, col3][i % 3]:
                inputs[column] = st.number_input(
                    column,
                    min_value=int(df[column].min()),
                    max_value=int(df[column].max()),
                    value=int(student[column])
                )

        submitted = st.form_submit_button(
            "🔮 Predict Final Grade",
            use_container_width=True
        )

    
    if submitted:
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("Prediction Result")

        st.metric("Predicted Final Grade", f"{prediction:.2f} / 20")

        if prediction >= 10:
            status = "Pass"
            st.success("The predicted grade is a pass.")
        else:
            status = "Below passing threshold"
            st.warning("The predicted grade is below the passing threshold of 10.")

        # Create individual student report
        report = input_df.copy()
        report["Predicted_Final_Grade"] = round(float(prediction), 2)
        report["Prediction_Status"] = status
        report["Model"] = "Random Forest"

        # Convert report to CSV
        csv_report = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Student Prediction Report",
            data=csv_report,
            file_name="individual_student_prediction.csv",
            mime="text/csv",
            use_container_width=True
        )


elif page == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Total Features", len(df.columns) - 1)
    col3.metric("Average Final Grade", round(df["G3"].mean(), 2))

    st.subheader("Student Dataset")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Final Grade Distribution")
    grade_counts = df["G3"].value_counts().sort_index()
    st.bar_chart(grade_counts)

elif page == "Model Performance":

    st.header("Machine Learning Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Algorithm", "Random Forest")
    col2.metric("MAE", "1.209")
    col3.metric("R² Score", "0.803")

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
        "MAE": [1.647, 1.278, 1.209],
        "R2 Score": [0.724, 0.733, 0.803]
    })

    st.dataframe(comparison, use_container_width=True)


    st.divider()
    st.subheader("Actual vs Predicted Grades")

    # Load saved prediction results
    results_path = BASE_DIR / "reports" / "prediction_results.csv"
    results = pd.read_csv(results_path)

    import matplotlib.pyplot as plt

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        results["Actual Grade"],
        results["Predicted Grade"],
        alpha=0.7
    )

    ax.plot([0, 20], [0, 20], linestyle="--")

    ax.set_xlabel("Actual Grade")
    ax.set_ylabel("Predicted Grade")
    ax.set_title("Actual vs Predicted Grades")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    # Prediction error distribution
    st.subheader("Prediction Error Distribution")

    results["Error"] = (
        results["Actual Grade"] - results["Predicted Grade"]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(results["Error"], bins=15, edgecolor="black")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Prediction Errors")
    ax.grid(axis="y")

    st.pyplot(fig)
    plt.close(fig)

    # Download prediction report
    st.divider()
    st.subheader("Download Prediction Report")

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv_data,
        file_name="student_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )