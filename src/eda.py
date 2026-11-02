import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from load_data import BASE_DIR

#load the dataset
BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "student-mat.csv", sep=";")

# 1. Grade distribution
plt.figure(figsize=(8,5))
sns.histplot(df['G3'], bins=10, kde=True)
plt.title('Distribution of final grades (G3)')
plt.xlabel('Final grade (G3)')
plt.ylabel('Number of Students ')
plt.savefig(BASE_DIR /"reports" / "grade_distribution.png")
plt.show()

# 2. Study time vs final grade
plt.figure(figsize=(8,5))
sns.boxplot(x='studytime',y='G3',data=df)
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade")
plt.savefig(BASE_DIR / "reports" / "studytime_vs_grade.png")
plt.show()

# 3. Previous failures vs final grade
plt.figure(figsize=(8, 5))
sns.boxplot(x="failures", y="G3", data=df)
plt.title("Previous Failures vs Final Grade")
plt.xlabel("Previous Failures")
plt.ylabel("Final Grade")
plt.savefig(BASE_DIR / "reports" / "failures_vs_grade.png")
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(10, 7))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
plt.title("Student Performance Correlation")
plt.savefig(BASE_DIR / "reports" / "correlation_heatmap.png")
plt.show()

print("EDA completed successfully!")
