# Elevvo ML Internship

Machine learning tasks completed during my Machine Learning internship at Elevvo.

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, imbalanced-learn

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
