from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import pandas as pd

# Load the model and feature order
model = joblib.load('best_xgboost_model.pkl')
feature_order = joblib.load('feature_order.pkl')

# Initialize FastAPI app
app = FastAPI(title="Titanic Survival Prediction API", description="Predict whether a passenger survived the Titanic disaster")

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to Titanic Survival Prediction API",
        "version": "1.0",
        "endpoints": {
            "docs": "http://127.0.0.1:8000/docs",
            "redoc": "http://127.0.0.1:8000/redoc",
            "predict": "POST /predict"
        }
    }

# Define the input data model using Pydantic
class PassengerInput(BaseModel):
    Pclass: int        # 1, 2, or 3
    Age: float         # e.g. 25.0
    SibSp: int         # siblings/spouses aboard
    Parch: int         # parents/children aboard
    Fare: float        # ticket fare
    Sex: str           # "male" or "female"
    Embarked: str      # "S", "Q", or "C"

# Define the API endpoint for prediction
@app.post("/predict")
# User input will be validated against the PassengerInput model
def predict_survival(input_data: PassengerInput):
    # Create a DataFrame from the input data 
    # Encode input data as per the training data encoding
    input_dict = {
        "Pclass": input_data.Pclass,
        "Age": input_data.Age,
        "SibSp": input_data.SibSp,
        "Parch": input_data.Parch,
        "Fare": input_data.Fare,
        "Sex_male": 1 if input_data.Sex == "male" else 0,
        "Embarked_Q": 1 if input_data.Embarked == "Q" else 0,
        "Embarked_S": 1 if input_data.Embarked == "S" else 0
    }

    # Match the order of features of input data with feature_order
    input_df = pd.DataFrame([input_dict])[feature_order]

    # Make prediction using the loaded model
    prediction = model.predict(input_df)[0]

    return {
        "Prediction": int(prediction),
        "Result":"Survived" if prediction ==1 else 'Not survived',
    }