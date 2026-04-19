#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression     
from sklearn.metrics import r2_score, mean_squared_error

#Load Dataset
df = pd.read_csv("dataset/ai_worker_burnout_attrition_2026.csv")
print(df.info())
print(df.describe())

print("First 5 entries:\n")
print(df.head())

# Feature Selection : Keeping only useful columns 

df = df[[
    "years_experience",
    "salary_usd_k",
    "remote_work_type",
    "team_size",
    "ai_tools_used_per_day",
    "hours_with_ai_assistance_daily",
    "ai_replaces_my_tasks_pct",
    "weekly_ai_upskilling_hrs",
    "productivity_score",
    "burnout_score",
    "job_satisfaction_1_5",
    "fear_of_ai_replacement",
    "attrition_risk"
]]
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
plt.figure(figsize=(12, 8))
sns.heatmap(
    df[num_cols].corr(),
    annot=True,
)
plt.title("Correlation Heatmap - All Numeric Variables")
plt.show()


# ============================================================
#  OBJECTIVE 1
#  Does the % of tasks replaced by AI increase burnout?
#  Graph: Scatter Plot
#  Columns: ai_replaces_my_tasks_pct vs burnout_score
# ============================================================

print("\n--- Objective 1: AI Task Replacement vs Burnout ---")

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="ai_replaces_my_tasks_pct",
    y="burnout_score",
    data=df,
    color="steelblue",
    alpha=0.6
)
plt.title("Objective 1 - AI Task Replacement % vs Burnout Score")
plt.xlabel("% of Tasks Replaced by AI")
plt.ylabel("Burnout Score")
plt.tight_layout()
plt.show()

# checking the actual correlation value
corr1 = df["ai_replaces_my_tasks_pct"].corr(df["burnout_score"])
print(f"Correlation between ai_replaces_my_tasks_pct and burnout_score: {corr1:.4f}")
# positive value means as AI replaces more tasks, burnout goes up

# ============================================================
#  OBJECTIVE 2
#  Does higher burnout lead to lower job satisfaction?
#  Graph: Scatter Plot
#  Columns: burnout_score vs job_satisfaction_1_5
# ============================================================

print("\n--- Objective 2: Burnout vs Job Satisfaction ---")

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="burnout_score",
    y="job_satisfaction_1_5",
    data=df,
    color="coral",
    alpha=0.6
)
plt.title("Objective 2 - Burnout Score vs Job Satisfaction")
plt.xlabel("Burnout Score")
plt.ylabel("Job Satisfaction (1-5)")
plt.tight_layout()
plt.show()

corr2 = df["burnout_score"].corr(df["job_satisfaction_1_5"])
print(f"Correlation between burnout_score and job_satisfaction_1_5: {corr2:.4f}")
# negative value means as burnout increases, satisfaction drops

# ============================================================
#  OBJECTIVE 3
#  Do employees with higher attrition risk have more AI
#  task replacement?
#  Graph: Boxplot
#  Columns: attrition_risk (grouped) vs ai_replaces_my_tasks_pct
# ============================================================

print("\n--- Objective 3: Attrition Risk vs AI Task Replacement ---")

plt.figure(figsize=(8, 5))
sns.boxplot(
    x="attrition_risk",
    y="ai_replaces_my_tasks_pct",
    order=["Low", "Medium", "High"],
    data=df,
    palette="Set2"
)
plt.title("Objective 3 - Attrition Risk vs AI Task Replacement %")
plt.xlabel("Attrition Risk Level")
plt.ylabel("% of Tasks Replaced by AI")
plt.tight_layout()
plt.show()

# printing the median for each group to see the difference clearly
print("Median AI task replacement % by attrition risk group:")
print(df.groupby("attrition_risk")["ai_replaces_my_tasks_pct"].median())
# expected: Low < Medium < High (step-up pattern)

# ============================================================
#  OBJECTIVE 4
#  Does spending more hours with AI assistance improve
#  productivity?
#  Graph: Scatter Plot
#  Columns: hours_with_ai_assistance_daily vs productivity_score
# ============================================================

print("\n--- Objective 4: Daily AI Hours vs Productivity ---")

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="hours_with_ai_assistance_daily",
    y="productivity_score",
    data=df,
    color="mediumseagreen",
    alpha=0.6
)
plt.title("Objective 4 - Daily AI Hours vs Productivity Score")
plt.xlabel("Hours with AI Assistance Daily")
plt.ylabel("Productivity Score")
plt.tight_layout()
plt.show()

corr4 = df["hours_with_ai_assistance_daily"].corr(df["productivity_score"])
print(f"Correlation between hours_with_ai_assistance_daily and productivity_score: {corr4:.4f}")
# positive means more AI assistance hours = higher productivity

# ============================================================
#  OBJECTIVE 5 - HYPOTHESIS TESTING
#  Does job satisfaction significantly differ between
#  High and Low attrition risk employees?
#  Test: Independent Samples T-Test
# ============================================================

print("\n--- Objective 5: Hypothesis Testing ---")

# separating the two groups
high_risk = df[df["attrition_risk"] == "High"]["job_satisfaction_1_5"]
low_risk  = df[df["attrition_risk"] == "Low"]["job_satisfaction_1_5"]

print(f"Mean satisfaction - High attrition group : {high_risk.mean():.3f}")
print(f"Mean satisfaction - Low attrition group  : {low_risk.mean():.3f}")
print(f"Difference in means                      : {low_risk.mean() - high_risk.mean():.3f}")

# running the t-test
# i used independent t-test because these are two different groups
# of different employees (not the same people measured twice)
t_stat, p_val = ttest_ind(high_risk, low_risk)

print("\nH0 (Null Hypothesis)     : There is no significant difference in job")
print("                           satisfaction between High and Low attrition risk")
print("H1 (Alternate Hypothesis): High attrition employees have significantly lower")
print("                           job satisfaction than Low attrition employees")

print(f"\nT-Statistic : {t_stat:.4f}")
print(f"P-Value     : {p_val:.8f}")

# if p < 0.05 we reject H0
if p_val < 0.05:
    print("\nResult: Reject H0")
    print("Conclusion: There IS a significant difference in job satisfaction.")
    print("            Dissatisfied employees are significantly more likely to leave.")
else:
    print("\nResult: Fail to Reject H0")
    print("Conclusion: No significant difference found.")

# ============================================================
#  OBJECTIVE 6 - SIMPLE LINEAR REGRESSION
#  Can we predict burnout_score using only ai_replaces_my_tasks_pct?
#  Model: Simple Linear Regression (1 input, 1 output)
#  X (independent) : ai_replaces_my_tasks_pct
#  Y (dependent)   : burnout_score
#
#  i chose ai_replaces_my_tasks_pct as the predictor because
#  it had the strongest correlation with burnout (r = 0.61)
#  out of all the columns in the dataset
# ============================================================
 
print("\n--- Objective 6: Simple Linear Regression ---")
print("Predicting burnout_score using ai_replaces_my_tasks_pct")
 
# selecting only 1 feature (X) and 1 target (y)
X = df[["ai_replaces_my_tasks_pct"]]   # double brackets because sklearn needs 2D array
y = df["burnout_score"]
 
# splitting into 80% training and 20% testing
# random_state=42 makes sure we get the same split every time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
print(f"Training set size : {X_train.shape[0]} rows")
print(f"Testing set size  : {X_test.shape[0]} rows")
 
# training the simple linear regression model
# it finds the best fit line: burnout = m * ai_replaces_pct + c
model = LinearRegression()
model.fit(X_train, y_train)
 
# using the trained model to predict burnout on test data
y_pred = model.predict(X_test)
 
# --- model equation ---
slope     = model.coef_[0]
intercept = model.intercept_
 
print(f"\nRegression Equation:")
print(f"  burnout_score = {slope:.4f} * ai_replaces_my_tasks_pct + {intercept:.4f}")
print(f"\nSlope (m)     : {slope:.4f}")
print(f"  meaning: for every 1% increase in AI task replacement,")
print(f"           burnout score increases by {slope:.2f} points")
print(f"Intercept (c) : {intercept:.4f}")
print(f"  meaning: when AI replaces 0% of tasks, predicted burnout is {intercept:.2f}")
 
# --- evaluation metrics ---
r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
 
print(f"\nR2 Score : {r2:.4f}  (model explains {r2*100:.1f}% of variance in burnout)")
print(f"MSE      : {mse:.4f}  (average squared prediction error)")
print(f"RMSE     : {rmse:.4f}  (on average prediction is off by {rmse:.2f} burnout points)")
 
# --- scatter plot with regression line ---
# this shows the actual data points AND the line the model learned
plt.figure(figsize=(8, 6))
 
# plot the actual data points
plt.scatter(
    X_test, y_test,
    color="steelblue", alpha=0.5, label="Actual Data"
)
 
# plot the regression line on top
plt.plot(
    X_test, y_pred,
    color="red", linewidth=2, label="Regression Line"
)
 
plt.xlabel("AI Task Replacement % (ai_replaces_my_tasks_pct)")
plt.ylabel("Burnout Score")
plt.title(f"Objective 6 - Simple Linear Regression\n"
          f"burnout = {slope:.2f} × ai_replaces_pct + {intercept:.2f}  |  R² = {r2:.2f}")
plt.legend()
plt.tight_layout()
plt.show()
 
# ============================================================
#  PROJECT COMPLETED
# ============================================================
 
print("\nProject completed successfully.")
print("Summary of findings:")
print(f"  Obj 1 - AI task replacement vs burnout    : r = {corr1:.2f} (strong positive)")
print(f"  Obj 2 - Burnout vs job satisfaction       : r = {corr2:.2f} (strong negative)")
print(f"  Obj 3 - Attrition risk vs AI replacement  : clear step-up in boxplot medians")
print(f"  Obj 4 - AI hours vs productivity           : r = {corr4:.2f} (moderate positive)")
print(f"  Obj 5 - Hypothesis test (t-test)           : p = {p_val:.6f} -> Reject H0")
print(f"  Obj 6 - Simple linear regression           : R2 = {r2:.2f}")
 