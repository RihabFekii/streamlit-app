import requests
import streamlit as st


# User input: ID for the GET request to the Context Broker (to get the entities)
def user_input(default_id):
	return  st.text_input("User input", default_id)


# GET attributes from the Context broker corresponding to id
def get_attributes(url):

	#url="http://backend.docker:8000/get_entities/" + id

	headers = {
	'Link': '<http://context/water-ngsi.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
	'Accept': 'application/ld+json'
	}

	entities = requests.request("GET",url,headers=headers)

	st.write(entities.json())
	att = entities.json()  

	return att


# Extracts attributes from the GET entities request payload, which will be injected to the ML model 
def extract_attributes(att):
		
	result = dict(Ph=att['Ph'],Hardness=att['Hardness'], Solids=att['Solids'],Chloramines=att['Chloramines'], 
	Sulfate=att['Sulfate'], Conductivity=att['Conductivity'], Organic_carbon=att['Organic_carbon'],
	Trihalomethanes=att['Trihalomethanes'], Turbidity=att['Turbidity'])

	return result 
