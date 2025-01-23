from fastapi import FastAPI
from endpoints import crocodile_api

app = FastAPI()

# Include the crocodile router
app.include_router(crocodile_api.router)

@app.get("/")
def read_root():
    return {"message": "Hello World"}