from fastapi import FastAPI , Request , Body
from pydantic import BaseModel
import pickle
from fastapi.logger import logger
import httpx
import json
#import requests

app = FastAPI()

#Creating a class for the attributes input to the ML model.
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


#Loading the trained model
with open("./finalized_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

#Sending a post request to the “/prediction” route with a request body. 
# The request body contains the key-value pairs of the water metrics parameters
# We should expect a JSON response with the potability classified.

#Columns are: ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity','Organic_carbon', 'Trihalomethanes']
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
    print("Hiii")
    print(data)


#Query to the Context Broker to get entities 
#url1="http://orion.docker:1027/ngsi-ld/v1/entities/urn:ngsi-ld:WaterPotabilityMetrics:001?options=keyValues"

@app.get("/get_entities/{id}")
async def get_entities(id:str ):
    #requires_response_body = True
    url="http://orion.docker:1027/ngsi-ld/v1/entities/" +  id + "?options=keyValues"

    headers = {
	'Link': '<http://context/water-ngsi.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
	'Accept': 'application/ld+json'
	}

    client = httpx.Client()
    response = client.get(url,headers=headers)
    
    logger.info(response.json())
    return response.json()

@app.patch("/prediction/{id}/{potability}")
def notify_prediction(id:str,potability:str):
    url = "http://orion.docker:1027/ngsi-ld/v1/entities/" + id + "/attrs/Potability"

    payload = json.dumps({
	"value": potability,
	"type": "Property"
	})
    headers = {
	'Content-Type': 'application/json',
	'Link': '<http://context/water-ngsi.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
	}
    client = httpx.Client()
    response = client.patch(url, headers=headers, data=payload)

    return response.json()




# homepage route
@app.get("/")
def read_root():
	return {'message': 'This is the homepage of the API '}



