# 🚢 Titanic Survival Prediction using XGBoost & LightGBM

## 📌 Project Overview

This project focuses on predicting passenger survival on the Titanic using machine learning. The goal is not just to build a model, but to compare two powerful algorithms — **XGBoost** and **LightGBM** — and understand which performs better and why.

---

## 🎯 Objective

The objective of this project is to:

* Predict whether a passenger survived the Titanic disaster
* Compare the performance of XGBoost and LightGBM
* Evaluate models using multiple metrics
* Understand the key factors influencing survival

---

## 📊 Dataset

* Source: Titanic Dataset (Kaggle)
* Total records: 891 passengers
* Key features:

  * Passenger class (Pclass)
  * Gender (Sex)
  * Age
  * Fare
  * Family information (SibSp, Parch)

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to uncover patterns in the dataset.

### Key Insights:

* Female passengers had significantly higher survival rates
* First-class passengers were more likely to survive
* Passengers who paid higher fares had better survival chances
* Age had some influence, but was not the strongest factor

These insights helped guide preprocessing and model building.

---

## ⚙️ Data Preprocessing

To prepare the data for modeling:

* Missing values were handled:

  * Age → filled using median
  * Embarked → filled using mode
* Irrelevant columns were removed:

  * Name, Ticket, Cabin, PassengerId
* Categorical variables were encoded using one-hot encoding

These steps ensured clean and model-ready data.

---

## 🤖 Models Used

Two gradient boosting algorithms were implemented:

* **XGBoost Classifier**
* **LightGBM Classifier**

Both models were trained on the same dataset to ensure a fair comparison.

---

## 📈 Model Evaluation

| Model    | Accuracy | Precision | Recall | F1 Score |
| -------- | -------- | --------- | ------ | -------- |
| XGBoost  | 79.89%   | 77.97%    | 66.67% | 71.88%   |
| LightGBM | 78.21%   | 76.79%    | 62.32% | 68.80%   |

### 🔍 Interpretation

XGBoost slightly outperforms LightGBM across all metrics, especially in recall and F1-score, indicating better ability to identify surviving passengers.

---

## 🔁 Cross Validation

* 5-Fold Cross Validation Accuracy: **~82%**

This shows that both models are stable and perform consistently across different data splits.

---

## ⚡ Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** to improve model performance.

* Tuned parameters:

  * n_estimators
  * max_depth
  * learning_rate

Due to the relatively small and simple nature of the dataset, tuning resulted in only marginal improvements, indicating that the base models were already performing well.

---

## 📊 Feature Importance

Feature importance analysis was performed using the tuned models.

### Key Features:

* Gender (Sex)
* Passenger Class (Pclass)
* Fare

These features had the highest impact on survival prediction, highlighting the role of socio-economic and demographic factors.

---

## 📊 Visualization

* Bar chart comparing model performance
* Feature importance plots

Visualizations were used to clearly compare models and interpret results.

---

## 📌 Comparative Analysis

| Model    | Accuracy | Precision | Recall | F1 Score |
| -------- | -------- | --------- | ------ | -------- |
| XGBoost  | 79.89%   | 77.97%    | 66.67% | 71.88%   |
| LightGBM | 78.21%   | 76.79%    | 62.32% | 68.80%   |

### 🔍 Insights:

* XGBoost performs slightly better across all metrics
* Both models show strong and consistent performance
* LightGBM is more efficient but slightly less accurate

---

## 🏁 Conclusion

Both XGBoost and LightGBM demonstrated strong performance in predicting Titanic survival. However, XGBoost achieved slightly better results across all evaluation metrics. The consistency between cross-validation and test results indicates that the models generalize well and are not overfitting.

---

## 🚀 Future Improvements

While the current model performs well, there is always room for improvement.

In future iterations, the project can be enhanced by exploring more advanced feature engineering techniques, such as extracting meaningful information from passenger names (titles like Mr, Mrs, etc.) or creating new features based on family size.

Additionally, applying more advanced hyperparameter tuning techniques and experimenting with other machine learning models could further improve performance.

To make the project more practical and impactful, the model can be deployed using tools like Streamlit, allowing users to interact with it and make real-time predictions.

Finally, working with larger and more complex real-world datasets would help build more robust models and better reflect real-world scenarios.

---

## 💻 Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost, LightGBM

---

## 📌 Author

Chaitanya Krishna
