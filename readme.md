# Prototype your data sceince projects easily with Streamlit and FastAPI


The goal of thos project is to enable a user to easily preprocess data, train and test an ML model via a web interface.  
The data is provisioned in real-time via REST API and by means of deploying the FIWARE Context Broker.

The use case of this project is water potability classification based on water metrics. 

## Overview 
This is a simple web interface which showcases the prediction phase: 

![Streamlit_App](https://miro.medium.com/max/1400/1*2lBlL4ltEz-lxRAnpP_neg.png)

## Frontend app with Streamlit
The web app is composed of multiple tabs/interfaces that achieve the different Machine Learning steps manually by the user, relying on the pre-built UI components (e.g for data pre-processing, for configuring the ML model parameters, training,...) 

You can see the app in the browser using this URL: 
http://localhost:8501/

## Backend app with FASTAPI 
FastAPI is a python web framework which I used to implement the APIs for this web application.  

You could check the documentation of the APIs by visiting this link: 
http://localhost:8000/docs


## Running the app with Docker-compose 

To package the whole solution which uses multiple images/containers, I used Docker Compose. 

Since we have multiple containers communcating with each other, I created a bridge network called AIservice. 

First create the network AIService by running this command:

    "docker netork create AIservice"


Run the whole application by executing this command:

    "docker-compose up -d --build"




