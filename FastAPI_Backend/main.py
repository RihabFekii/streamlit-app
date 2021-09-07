from fastapi import FastAPI , Request
from pydantic import BaseModel
import pickle
from fastapi.logger import logger
import httpx
import json

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
@app.post('/prediction' )
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
    return {'Prediction':  pred_name}

# ph : float, Hardness :float ,Solids : float, Chloramines : float, Sulfate : float, Conductivity : float, Organic_carbon : float, Trihalomethanes : float, Turbidity : float
#to get data from context broker or query 
@app.post('/extract_attributes')
def ML_model_input(): 
    p= dict()
    ph = p['Ph']
    Hardness = p['Hardness']
    Solids = p['Solids']
    Chloramines = p['Chloramines']
    Sulfate = p['Sulfate']
    Conductivity = p['Conductivity']
    Organic_carbon = p['Organic_carbon']
    Trihalomethanes = p['Trihalomethanes']
    Turbidity = p['Turbidity']
    
    result = water_metrics()
    result['Ph']=ph
    result['Hardness']= Hardness
    result['Solids']=Solids
    result['Chloramines']= Chloramines
    result['Sulfate']=Sulfate
    result['Conductivity']= Conductivity
    result['Organic_carbon']=Organic_carbon
    result['Trihalomethanes']= Trihalomethanes
    result['Turbidity']= Turbidity

    return result

#Query to the Context Broker to get entities 
#url1="http://orion.docker:1027/ngsi-ld/v1/entities/urn:ngsi-ld:WaterPotabilityMetrics:001?options=keyValues"

@app.get("/get_entities/{id}")
async def get_entities(id:str ):
    #requires_response_body = True
    url="http://orion.docker:1027/ngsi-ld/v1/entities/" +  id + "?options=keyValues"
    client = httpx.Client()
    response = client.get(url)
    
    logger.info(response.json())
    return response.json()


# homepage route
@app.get("/")
def read_root():
	return {'message': 'This is the homepage of the API '}

@app.post('/data')
def show_data(data: water_metrics):
    return {'message': data}

