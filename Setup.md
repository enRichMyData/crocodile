# Setup Guide

## Start Docker Container
- First start Docker Desktop App
```bash
docker-compose up -d --build
```

--------------------------------------------
## Testing

### Onboard Data to MongoDB and Start Entitylinking
```bash
docker exec -it crocodile_jupyter bash
```
- Inside the container, run the following command
```bash
cd test
```
- This command will onboard data to MongoDB
```bash
python3 onboard_data_test.py
```
- This command will start the entitylinking process
```bash
python3 start_computation.py
```
```bash
exit
```

### Access MongoDB
```bash
docker exec -it crocodile_mongodb mongosh
```
- Inside the shell, run the following command
```bash
use crocodile_db
```
- To see the tables
```bash
show collections
```
- To see the data in a table
```bash
db.input_data.find().pretty()
```
```bash
exit
```

--------------------------------------------
## Evnironment Variables
```bash
ENTITY_RETRIEVAL_ENDPOINT=# Endpoint for entity retrieval
ENTITY_BOW_ENDPOINT=# Endpoint for entity bag of words
ENTITY_RETRIEVAL_TOKEN=# Token for entity retrieval

MONGO_URI=# MongoDB URI
MONGO_SERVER_PORT=# MongoDB Server Port
MONGO_VERSION=# MongoDB Version

FASTAPI_APP_NAME=# FastAPI App Name
FASTAPI_SERVER_PORT=# FastAPI Server Port

JUPYTER_SERVER_PORT=# Jupyter Server Port

DB_NAME=# Database Name
DEBUG=# Debug Mode
```
