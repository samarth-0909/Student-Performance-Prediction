# Student Performance Prediction & Analytics System

A Data Science and Machine Learning project that explores student academic performance and aims to predict student outcomes using Python.

## Project Overview

The **Student Performance Prediction & Analytics System** is designed to analyse students' academic performance, identify important factors affecting final grades, and develop machine learning models to predict student performance.

The project uses the Student Performance dataset from the UCI Machine Learning Repository.

## Objectives

- Analyse student academic performance.
- Explore factors that influence final grades.
- Visualise patterns and relationships in student data.
- Develop machine learning models to predict final grades.
- Identify students who may be at academic risk.
- Build an interactive dashboard for analysing predictions.

*Note: The machine learning and dashboard features are planned development stages.*

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (planned)
- Streamlit (planned)
- Git and GitHub

## Dataset

**Dataset:** Student Performance Data Set

**Source:** UCI Machine Learning Repository

[Student Performance Dataset – UCI](https://archive.ics.uci.edu/dataset/320/student+performance)

The dataset contains student demographic, social, school-related, and academic information.

The mathematics dataset (`student-mat.csv`) is currently used in this project.

## Exploratory Data Analysis (EDA)

The initial exploratory data analysis includes:

1. Loading and inspecting the dataset.
2. Checking data types and missing values.
3. Analysing final grade distribution.
4. Exploring study time versus final grades.
5. Analysing previous failures versus final grades.
6. Creating a correlation heatmap.

### Visualisations

The following visualisations have been generated:

- Final Grade Distribution
- Study Time vs Final Grade
- Previous Failures vs Final Grade
- Correlation Heatmap

## Project Structure

```text
Student-Performance-Prediction/
│
├── data/
│   └── student-mat.csv
│
├── notebooks/
│
├── src/
│   ├── load_data.py
│   └── eda.py
│
├── models/
│
├── reports/
│   ├── correlation_heatmap.png
│   ├── failures_vs_grade.png
│   ├── grade_distribution.png
│   └── studytime_vs_grade.png
│
├── screenshots/
│
├── .gitignore
└── README.md
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/samarth-0909/Student-Performance-Prediction.git
```

### 2. Open the project folder

```bash
cd Student-Performance-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 6. Run the data loading script

```bash
python src/load_data.py
```

### 7. Run the exploratory data analysis

```bash
python src/eda.py
```

## Current Project Status

- [x] Project setup
- [x] Dataset collection
- [x] Data loading
- [x] Exploratory Data Analysis
- [x] Data visualisations
- [ ] Data preprocessing
- [ ] Feature engineering
- [ ] Machine Learning model development
- [ ] Model evaluation and comparison
- [ ] Student performance prediction
- [ ] Streamlit dashboard
- [ ] Final project documentation

## Future Improvements

- Compare multiple machine learning algorithms.
- Predict student final grades.
- Develop pass/fail classification.
- Identify students who may need academic support.
- Build an interactive Streamlit dashboard.

## Author

**Samarth Bhat**

GitHub: [@samarth-0909](https://github.com/samarth-0909)

---

This project is being developed as part of my Third Year (TY) Data Science portfolio.
