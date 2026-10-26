# Elevvo ML Internship

Machine learning tasks completed during my Machine Learning internship at Elevvo.

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

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
