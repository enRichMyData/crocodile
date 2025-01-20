# Setup Guide

## Setup Environment
- Make sure you have `docker` installed
- Make a .env file in the root directory
- Add the following variables to the .env file
```
ENTITY_RETRIEVAL_ENDPOINT=SOME_URL
ENTITY_BOW_ENDPOINT=SOME_URL
ENTITY_RETRIEVAL_TOKEN=SOME_TOKEN
MONGO_URI=mongodb://mongodb:27017
MONGO_SERVER_PORT=27017
JUPYTER_SERVER_PORT=8888
MONGO_VERSION=7.0
```

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