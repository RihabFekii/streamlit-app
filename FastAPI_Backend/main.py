from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

#Creating a class for the attributes input to the ML model.
class water(BaseModel):
	ph : float
	Hardness :float
	Solids : float
	Chloramines : float
	Sulfate : float
	Conductivity : float
	Organic_carbon : float
	Trihalomethanes : float
	Turbidity : float

#Loading the trained model
with open("./FastAPI_Backend/finalized_model.pkl", "rb") as f:
    model = pickle.load(f)

# Hello World route
@app.get("/")
def read_root():
	return {'message': 'This is the homepage of the API '}

@app.post('/data')
def show_data(data: water):
    return {'message': data}

