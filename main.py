from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd

# Create an instance of the FastAPI class
app = FastAPI()

# Load the pre-trained models
forest_pipeline = joblib.load("./models/random_forest_model.joblib")
svm_model = joblib.load("./models/support_vector_machine_model.joblib")
logistic_regression_model = joblib.load("./models/logistic_regression_model.joblib")
encoder = joblib.load("./models/label_encoder.joblib")

# Define a request body model using Pydantic
class PatientData(BaseModel):
    PRG: int
    PL: int
    PR: int
    SK: int
    TS: int
    M11: float
    BD2: float
    Age: int
    Insurance: int

# Define a route for the root endpoint "/"
@app.get("/")
def read_root():
    return {"message": "Welcome to the Patient Data API"}

def get_highest_prob_class(probs, encoder):
    # Get the class with the highest probability
    max_prob_index = probs.argmax()
    return encoder.inverse_transform([max_prob_index])[0]

# Define a route to handle POST requests at "/predict_random_forest" endpoint
@app.post("/predict_random_forest")
def random_forest_prediction(data: PatientData):
    # Convert the data to DataFrame
    df = pd.DataFrame([data.dict()])  
    
    # Perform prediction using the Random Forest model
    predictions_rf = forest_pipeline.predict(df)
    probs_rf = forest_pipeline.predict_proba(df)
    
    # Get the class with the highest probability
    predicted_class_rf = get_highest_prob_class(probs_rf, encoder)
    
    return {'predicted_class_random_forest': predicted_class_rf}

# Define a route to handle POST requests at "/predict_svm" endpoint
@app.post("/predict_svm")
def svm_prediction(data: PatientData):
    # Convert the data to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Perform prediction using the SVM model
    predictions_svm = svm_model.predict(df)
    probs_svm = svm_model.predict_proba(df)
    
    # Get the class with the highest probability
    predicted_class_svm = get_highest_prob_class(probs_svm, encoder)
    
    return {'predicted_class_svm': predicted_class_svm}

# Define a route to handle POST requests at "/predict_logistic_regression" endpoint
@app.post("/predict_logistic_regression")
def logistic_regression_prediction(data: PatientData):
    # Convert the data to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Perform prediction using the Logistic Regression model
    predictions_lr = logistic_regression_model.predict(df)
    probs_lr = logistic_regression_model.predict_proba(df)
    
    # Get the class with the highest probability
    predicted_class_lr = get_highest_prob_class(probs_lr, encoder)
    
    return {'predicted_class_logistic_regression': predicted_class_lr}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8080, debug=True)
