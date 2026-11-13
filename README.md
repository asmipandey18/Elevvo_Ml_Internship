# Elevvo ML Internship

Machine learning tasks completed during my Machine Learning internship at Elevvo.

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, imbalanced-learn, XGBoost, statsmodels

## Task 1: Student Score Prediction
[Open the notebook](task1_student_score_prediction/Student_score_prediction.ipynb)

**Goal:** Predict students' exam scores from their study hours and other factors.

**Dataset:** Student Performance Factors (Kaggle)

**What I did:**
- Cleaned the data: removed duplicates, filled missing values, and removed an invalid exam score above 100
- Explored the data with scatter plots, a histogram, and a correlation heatmap
- Split the data into 80% training and 20% testing
- Trained a Linear Regression model on study hours
- Bonus: compared it with Polynomial Regression and tested different feature combinations

**Results (test set):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (study hours) | 2.42 | 3.16 | 0.2469 |
| Polynomial Regression (degree 2) | 2.42 | 3.15 | 0.2479 |
| Study + Attendance + Previous Scores | 1.33 | 2.13 | 0.6554 |
| Multiple features (6 numeric) | 1.24 | 2.04 | 0.6843 |

**Key findings:** Study hours alone explain only about 25% of the variation in scores. Polynomial regression barely helped, and adding more features improved R² a lot, to 0.68.

## Task 4: Loan Approval Prediction
[Open the notebook](task4_loan_approval_prediction/Loan_approval_prediction.ipynb)

**Goal:** Predict whether a loan application will be approved or rejected.

**Dataset:** Loan Approval Prediction Dataset (Kaggle), 4,269 applications. The data is mildly imbalanced (62% Approved, 38% Rejected).

**What I did:**
- Cleaned the column names and text values, checked for missing values and duplicates (none found), and encoded the categorical columns
- Explored the data with class balance, distribution, boxplot, approval-rate and correlation plots
- Split the data 80/20 with stratification
- Trained Logistic Regression and a Decision Tree, and evaluated them with precision, recall and F1
- Bonus: handled class imbalance with class weights and SMOTE (training data only)
- Extra: 5-fold cross-validation to confirm the results are stable

**Results (test set):**

| Model | Accuracy | Rejected Recall | Rejected F1 |
|---|---|---|---|
| Logistic Regression | 0.913 | 0.867 | 0.883 |
| LR + Class Weights | 0.924 | 0.929 | 0.902 |
| LR + SMOTE | 0.919 | 0.910 | 0.895 |
| Decision Tree | 0.981 | 0.975 | 0.975 |
| Tree + Class Weights | 0.977 | 0.963 | 0.969 |
| Tree + SMOTE | 0.980 | 0.975 | 0.974 |

**Key findings:** The Decision Tree performed best (cross-validated F1 of 0.979 vs 0.934 for Logistic Regression). CIBIL score was by far the most important feature. Class weights and SMOTE helped Logistic Regression's recall on rejected loans but made little difference to the tree.

## Task 7: Sales Forecasting
[Open the notebook](task7_sales_forecasting/Sales_forecasting.ipynb)

**Goal:** Forecast weekly sales for 45 Walmart stores from historical sales data.

**Dataset:** Walmart Dataset (Kaggle), 6,435 rows covering 143 weeks (Feb 2010 to Oct 2012).

**What I did:**
- Cleaned the data: checked for missing values, duplicates and negative sales (none found), converted the date, and sorted by store and date
- Explored total sales over time, holiday vs normal weeks, sales by store, and correlations
- Created time features (year, month, week, day), lag features (1, 2, 4 and 52 weeks) and a 4-week rolling average, all calculated within each store using only past values
- Split the data by date (train up to June 2012, test June to October 2012) to avoid leaking the future
- Trained Linear Regression and Random Forest, and compared them with a naive baseline (last week's sales)
- Plotted actual vs predicted sales over time
- Bonus: seasonal decomposition, and XGBoost with time-aware validation (TimeSeriesSplit)

**Results (test set):**

| Model | MAE | RMSE |
|---|---|---|
| Baseline (last week) | 52,681 | 80,073 |
| Linear Regression | 50,200 | 73,027 |
| Random Forest | 39,583 | 64,505 |
| XGBoost | 41,083 | 65,032 |

**Key findings:** Random Forest performed best, with an MAE of about 3.8% of average weekly sales, followed closely by XGBoost. Seasonal decomposition showed a strong yearly pattern with peaks around December and an almost flat long-term trend. Holiday_Flag does not mark the biggest pre-Christmas weeks, so month, week and lag_52 features matter more. The test period does not include the November and December peaks.

## Task 9: Industrial Predictive Maintenance
[Open the notebook](task9_predictive_maintenance/Predictive_maintenance.ipynb)

**Goal:** Predict machine failures from sensor readings while keeping false alarms low, because a false alarm stops production for nothing. Models were judged by the False Discovery Rate (false alarms ÷ all alarms) alongside recall, not by accuracy.

**Dataset:** AI4I 2020 Predictive Maintenance Dataset, 10,000 machine records. Only 3.4% are failures, so the data is severely imbalanced.

**What I did:**
- Cleaned the data, renamed columns, dropped ID columns, and excluded the failure-type columns from the features to avoid data leakage
- Explored class imbalance, sensor readings for normal vs failed machines, failure rate by product quality, and correlations
- Engineered features: temperature difference, power (torque × speed), and wear × torque
- Split the data 80/20 with stratification, and tuned the decision threshold using cross-validation on the training set only
- Compared Random Forest and XGBoost, including cost-sensitive (weighted) versions
- Trained one model per failure type, and compared against an unsupervised Isolation Forest

**Results (test set, 68 failures):**

| Model | False alarms | Missed failures | FDR |
|---|---|---|---|
| Random Forest (tuned) | 2 | 13 | 0.035 |
| XGBoost | 4 | 14 | 0.069 |
| XGBoost (weighted) | 20 | 11 | 0.260 |
| Isolation Forest (unsupervised) | 43 | 52 | 0.729 |

**Key findings:** The tuned Random Forest had the lowest false alarm rate and caught 55 of 68 failures. Torque was the strongest single sensor, and the engineered features were more informative than the raw sensors. Power and Overstrain failures were predicted perfectly and Heat Dissipation well, but Tool Wear could not be detected without many false alarms. The test set is small (68 failures), so results are only a rough guide.
