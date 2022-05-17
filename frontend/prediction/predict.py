import json

import requests


# ML model prediction function using the prediction API request
def predict(result): 
	 
	header = {'Content-Type': 'application/json'}
	url3= "http://backend.docker:8000/prediction"
	#url= "http://backend_aliases:8000/prediction"

	payload=json.dumps(result)
	
	response = requests.request("POST", url3, headers=header, data=payload)
	response = response.text

	return response


#notify context broker of Prediction attribute (Potability) update 
def notify_pred(id: str , resp:str):
	#url = "http://backend.docker:8000/patch_prediction/" + id
	url_notification= "http://backend.docker:8000/prediction/" + id + "/" + resp

	response = requests.request("PATCH", url_notification)

	return response.text
