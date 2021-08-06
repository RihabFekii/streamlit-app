from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

#Creating a class for the attributes input to the ML model.
class water_metrics(BaseModel):
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
with open("./finalized_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

#Sending a post request to the “/prediction” route with a request body. 
# The request body contains the key-value pairs of the water metrics parameters
# We should expect a JSON response with the potability classified.

#Columns are: ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity','Organic_carbon', 'Trihalomethanes']
@app.post('/prediction')
def get_potability(data: water_metrics):
    received = data.dict()
    ph = received['ph']
    Hardness = received['Hardness']
    Solids = received['Solids']
    Chloramines = received['Chloramines']
    Sulfate = received['Sulfate']
    Conductivity = received['Conductivity']
    Organic_carbon = received['Organic_carbon']
    Trihalomethanes = received['Trihalomethanes']
    Turbidity = received['Turbidity']
    pred_name = loaded_model.predict([[ph, Hardness, Solids,
                                Chloramines, Sulfate, Conductivity, Organic_carbon,
                                Trihalomethanes,Turbidity]]).tolist()[0]
    return {'prediction': pred_name}

@app.get('/prediction')
def potability(ph : float, Hardness :float ,Solids : float, Chloramines : float, Sulfate : float, Conductivity : float, Organic_carbon : float, Trihalomethanes : float, Turbidity : float):
	pred_name = loaded_model.predict([[ph, Hardness, Solids,Chloramines, Sulfate, Conductivity, 
	Organic_carbon,Trihalomethanes,Turbidity]]).tolist()[0]
	return {'prediction': pred_name}

# homepage route
@app.get("/")
def read_root():
	return {'message': 'This is the homepage of the API '}

@app.post('/data')
def show_data(data: water_metrics):
    return {'message': data}

