import json
import pickle

import httpx
from fastapi import Body, FastAPI
from fastapi.logger import logger
from pydantic import BaseModel

app = FastAPI()

# Creating a class for the attributes input to the ML model.
class WaterMetrics(BaseModel):
	Ph : float
	Hardness :float
	Solids : float
	Chloramines : float
	Sulfate : float
	Conductivity : float
	Organic_carbon : float
	Trihalomethanes : float
	Turbidity : float


# Loading the trained model
with open("./finalized_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Sending a post request to the “/prediction” route with a request body. 
# The request body contains the key-value pairs of the water metrics parameters
# We should expect a JSON response with the potability classified.

# Columns are: ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity','Organic_carbon', 'Trihalomethanes']
@app.post('/prediction' )
def get_potability(data: WaterMetrics):
    received = data.dict()
    ph = received['Ph']
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
    return  pred_name
    

@app.post('/subscription')
def subscription(data :dict = Body(...)):
    #output_data=await data.json()
    print(data)


@app.get("/get_entities/{id}")
async def get_entities(id:str ):
    
    url="http://orion.docker:1026/ngsi-ld/v1/entities/" +  id + "?options=keyValues"

    client = httpx.Client()
    response = client.get(url)
    
    logger.info(response.json())
    return response.json()

@app.patch("/prediction/{id}/{potability}")
def notify_prediction(id:str,potability:str):
    url = "http://orion.docker:1026/ngsi-ld/v1/entities/" + id + "/attrs/Potability"

    payload = json.dumps({
	"value": potability,
	"type": "Property"
	})
    headers = {
	'Content-Type': 'application/json',
	'Link': '<http://context/ngsi-context.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
	}
    client = httpx.Client()
    response = client.patch(url, headers=headers, data=payload)

    return response.json()


# homepage route
@app.get("/")
def read_root():
	return {'message': 'This is the homepage of the API '}



