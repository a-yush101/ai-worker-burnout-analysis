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