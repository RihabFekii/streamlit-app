## Build the Dockerfile

Paste in the terminal the following command: 
    
    "docker build -t  mystapp:latest  ."

To run the application: 

    "docker run -p 8501:8501 mystapp:latest" 

Then you can see the app in the brouser using the Network URL: http://localhost:8501/


## Backend build

docker build -t backend:latest .

docker run -p 8000:8000 backend:latest

check: http://localhost:8000/



## run the docker-compose

docker-compose up -d --build




