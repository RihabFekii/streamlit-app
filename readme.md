## Building the fontend

Paste in the terminal the following command: 
    
    "docker build -t  mystapp:latest  ."

To run the Streamlit application, use this command: 

    "docker run -p 8501:8501 mystapp:latest" 

Then you can see the app in the brouser using the Network URL: http://localhost:8501/


## building the backend 

    "docker build -t backend:latest ."

To run the FastAPI application, use this command:

    "docker run -p 8000:8000 backend:latest"

check: http://localhost:8000/ 

you could also have the documentation of your APIs using this link: 

http://localhost:8000/docs


## Docker-compose 

To package the whole solution which uses multiple images/service, I am using Docker Compose. 
So there will be no need to build each of the previous images( Streamlit and FastAPI) separately. 
In the docker-compose.yml file this is configured and you could so that by running this command:

    "docker-compose up -d --build"

If you have made some changes in your yml file configuration, you first need to stop your containers by: 

    "docker-compose down" 

Then to run again your application, use this command: 

    "docker-compose up -d"




