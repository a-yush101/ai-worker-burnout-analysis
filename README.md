# AI Worker Burnout & Attrition Risk Analysis

A Data Science project built using **Python** to analyze how AI adoption in workplaces impacts employee burnout, job satisfaction, productivity, and attrition risk.

---

## 📌 Project Overview

As Artificial Intelligence becomes more common in modern workplaces, organizations need to understand both its benefits and risks.

This project explores key questions such as:

- Does AI replacing human tasks increase burnout?
- Does burnout reduce job satisfaction?
- Does AI improve productivity when used as assistance?
- Are dissatisfied employees more likely to leave?
- Can burnout be predicted using AI adoption variables?

---

## 🎯 Objectives

### Objective 1: AI Task Replacement vs Burnout
Analyze whether a higher percentage of tasks replaced by AI increases employee burnout.

### Objective 2: Burnout vs Job Satisfaction
Study the relationship between burnout score and job satisfaction.

### Objective 3: Attrition Risk vs AI Replacement
Compare AI replacement levels across Low, Medium, and High attrition risk groups.

### Objective 4: AI Hours vs Productivity
Evaluate whether spending more hours with AI tools improves productivity.

### Objective 5: Hypothesis Testing (T-Test)
Test whether job satisfaction significantly differs between High and Low attrition risk employees.

### Objective 6: Linear Regression
Build a predictive model to estimate burnout score using AI task replacement percentage.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn

---

## 📊 Data Preprocessing

The dataset was cleaned using:

- Feature Selection
- Missing Value Removal
- Duplicate Removal
- Outlier Detection using Boxplots
- Outlier Removal using IQR Method

---

## 📈 Exploratory Data Analysis

Performed:

- Correlation Heatmap
- Scatter Plots
- Boxplots
- Trend Analysis

---

## 🔬 Statistical Testing

### Independent Samples T-Test

Compared job satisfaction between:

- High Attrition Risk Employees
- Low Attrition Risk Employees

### Result:

- p-value < 0.001
- Null Hypothesis Rejected

This means job satisfaction significantly differs between both groups.

---

## 🤖 Machine Learning Model

### Simple Linear Regression

Used:

- Input Variable: AI Task Replacement %
- Target Variable: Burnout Score

### Model Performance:

- R² Score = 0.38

This means 38% of burnout variation was explained by AI task replacement alone.

---

## 📌 Key Insights

- Higher AI task replacement is associated with higher burnout.
- Burnout strongly reduces job satisfaction.
- High attrition employees show higher AI replacement levels.
- AI assistance can improve productivity when used effectively.
- Job satisfaction is strongly linked with retention.

---

## 💡 Final Conclusion

AI works best when used as a supportive tool rather than a full replacement for employees.

Organizations should focus on:

- Human + AI collaboration
- Employee upskilling
- Burnout prevention
- Transparent automation strategies
- Data-driven HR decisions

---

## 📁 Project Files

```text
main.py
README.md
dataset.csv
Visualization/Images/
Visualization/main.html