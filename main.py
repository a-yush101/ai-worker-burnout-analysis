#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
df = pd.read_csv("dataset/ai_worker_burnout_attrition_2026.csv")
print(df.info())
print(df.describe())

print("First 5 entries:\n")
print(df.head())

#Keeping only useful columns
df = df[
    [
        "years_experience",
        "salary_usd_k",
        "remote_work_type",
        "team_size",
        "ai_tools_used_per_day",
        "hours_with_ai_assistance_daily",
        "weekly_ai_upskilling_hrs",
        "productivity_score",
        "burnout_score",
        "job_satisfaction_1_5",
        "fear_of_ai_replacement",
        "attrition_risk"
    ]
]

print(df.info())

print("First 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

#Null columns
print(df.isnull().sum()) #No Null values found
df = df.drop_duplicates()
#Outlier Detection
# Numeric Columns
num_cols = df.select_dtypes(include=np.number).columns

# Boxplots
for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

#Outlier Treatment
for col in num_cols:
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

#EDA

print("\nSummary Statistics:")
print(df.describe())

# Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#OBJECTIVE 1: Productivity Vs Burnout
sns.scatterplot(x="productivity_score", y="burnout_score", data=df)
plt.title("Productivity vs Burnout")
plt.show()
# This scatterplot shows how productivity changes with burnout.
# If points move upward together = positive relationship.
# If points move downward = negative relationship.
# If random spread = weak/no relationship.

corr1 = df["productivity_score"].corr(df["burnout_score"])
print("Correlation:", corr1)

# OBJECTIVE 2
# Study whether higher salary improves employee job satisfaction
sns.scatterplot(x="salary_usd_k", y="job_satisfaction_1_5", data=df)
plt.title("Salary vs Job Satisfaction")
plt.xlabel("Salary (USD in Thousands)")
plt.ylabel("Job Satisfaction (1 to 5)")
plt.show()
# This graph shows whether better paid employees report higher satisfaction.
