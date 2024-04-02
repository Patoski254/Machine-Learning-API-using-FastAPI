from fastapi import FastAPI
from pydantic import BaseModel

# Create an instance of the FastAPI class
app = FastAPI()

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

# Define a route to handle POST requests at "/predict" endpoint
@app.post("/predict")
def predict_sepsis(data: PatientData):
    # Perform prediction using the provided patient data
   
    prediction = {"Sepsis": "Positive" if data.PRG > 5 else "Negative"}
    return prediction
