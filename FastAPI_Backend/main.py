from fastapi import FastAPI

app = FastAPI()


# Hello World route
@app.get("/")
def read_root():
	return {"Hello": "World"}
