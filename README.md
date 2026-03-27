Titanic Survival Prediction using XGBoost & LightGBM
Project Overview
This project predicts passenger survival on the Titanic by comparing two powerful gradient boosting algorithms — XGBoost and LightGBM. The best performing model is served via a FastAPI application, containerized with Docker, and deployed live on Render.

Objective
Predict whether a passenger survived the Titanic disaster
Compare XGBoost and LightGBM across multiple evaluation metrics
Understand the key factors influencing survival
Deploy the best model as a live, publicly accessible FastAPI endpoint
Dataset
Source: Titanic Dataset (Kaggle) | Records: 891 passengers

Key features: Pclass, Sex, Age, Fare, SibSp, Parch

Exploratory Data Analysis
Key insights uncovered during EDA:

Female passengers had significantly higher survival rates
First-class passengers were more likely to survive
Higher fares correlated with better survival chances
Age had some influence but was not the strongest factor
Data Preprocessing
Age → filled using median | Embarked → filled using mode
Dropped: Name, Ticket, Cabin, PassengerId
Categorical encoding: one-hot encoding for Sex and Embarked
Models Used
XGBoost Classifier
LightGBM Classifier
Both trained on the same dataset for a fair comparison.

Model Evaluation
Model	Accuracy	Precision	Recall	F1 Score
XGBoost	79.89%	77.97%	66.67%	71.88%
LightGBM	78.21%	76.79%	62.32%	68.80%
XGBoost outperforms LightGBM across all metrics, particularly in recall and F1-score.

5-Fold Cross Validation Accuracy: ~82% — both models are stable and generalize well.

Hyperparameter Tuning
Tuned using GridSearchCV on n_estimators, max_depth, and learning_rate. Marginal improvements were observed, indicating the base models were already well-optimized for this dataset.

Feature Importance
Top features driving predictions: Sex, Pclass, Fare — reflecting the role of socio-economic and demographic factors in survival.

API Deployment
The XGBoost model is deployed as a FastAPI application and hosted live on Render.

Live API:

https://titanic-api-latest.onrender.com
Predict Endpoint: POST /predict

json
// Request
{
  "Pclass": 1,
  "Age": 25.0,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 72.5,
  "Sex": "female",
  "Embarked": "S"
}
// Response
{
  "Prediction": 1,
  "Result": "Survived"
}
Docker
Pull and run without any local setup:

bash
docker pull ck17041704/titanic-api
docker run -d -p 8000:8000 ck17041704/titanic-api
Open: http://localhost:8000/docs

Docker Hub: https://hub.docker.com/r/ck17041704/titanic-api

Project Structure
Titanic_Survival_Prediction/
├── Survival_prediction.ipynb   # EDA, training & evaluation
├── main.py                     # FastAPI application
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Dependencies
├── best_xgboost_model.pkl      # Trained model
├── feature_order.pkl           # Feature order for inference
├── Titanic_train.csv
└── Titanic_test.csv
Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, LightGBM, FastAPI, Uvicorn, Docker, Render

Conclusion
XGBoost achieved slightly better results across all metrics and was selected for deployment. The model generalizes well, as confirmed by cross-validation. The project is production-ready — served through FastAPI, containerized with Docker, and live on Render.

Future Improvements
Feature engineering from passenger names (titles: Mr, Mrs, etc.)
Advanced tuning with Optuna or Bayesian optimization
Streamlit frontend for non-technical users
Model monitoring and prediction logging on the deployed API
Author
Chaitanya Krishna
