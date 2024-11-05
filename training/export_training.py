# Import necessary modules
from pymongo import MongoClient
import pandas as pd
from tqdm import tqdm
import os

# MongoDB connection
client = MongoClient("mongodb://mongodb:27017/")
db = client["crocodile_db"]
collection = db["training_data"]
datasets = ["Round1_T2D", "Round3_2019", "2T_2020", "Round4_2020", "HardTablesR2", "HardTablesR3"]

# Create the output directory if it does not exist
output_dir = "training_data"
os.makedirs(output_dir, exist_ok=True)

buffer_size = 1000  # Define the buffer size for writing data incrementally

for dataset in datasets:
    # Set the output file path for the current dataset
    output_file = os.path.join(output_dir, f"{dataset}_training_samples.csv")
    
    # Clear the output file if it already exists to start fresh
    if os.path.exists(output_file):
        os.remove(output_file)

    documents = collection.find({"datasetName": dataset})
    total_docs = collection.count_documents({"datasetName": dataset})
    cea_gt = pd.read_csv(f"./Datasets/{dataset}/gt/cea.csv", header=None)
    
    # Create a dictionary to map (tableName, idRow, idCol) to QID
    qid_dict = {
        (row[0], row[1], row[2]): row[3].split('/')[-1]  # Extract only the QID part from the URL
        for _, row in cea_gt.iterrows()
    }
    
    buffer = []  # Initialize buffer for storing training samples

    # Process documents and fill the buffer
    group = 0
    for doc in tqdm(documents, total=total_docs):
        table_name = doc["tableName"]
        id_row = doc["idRow"]
        candidates = doc["candidates"]
        
        for id_col in candidates:
            # Get the QID from the dictionary
            qid = qid_dict.get((table_name, id_row + 1, int(id_col)))
            if qid is None:
                continue
            for candidate in candidates[id_col]:
                target = 1 if candidate["id"] == qid else 0
                key = f"{table_name} {id_row} {id_col}"
                temp = {"tableName": table_name, "key": key, "id": candidate["id"], "group": group}
                candidate["features"]["NE_match"] = 1 if candidate["features"]["NERtype"] == candidate["features"]["column_NERtype"] else 0
                sample = {**temp, **candidate["features"], **{"target": target}}
                buffer.append(sample)
            group += 1
        
        # Write the buffer to the CSV file if it reaches the defined size
        if len(buffer) >= buffer_size:
            df = pd.DataFrame(buffer)
            df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            buffer.clear()  # Clear the buffer after writing
    
    # Write any remaining data in the buffer to the file
    if buffer:
        df = pd.DataFrame(buffer)
        df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

    print(f"Training samples for {dataset} have been written to {output_file}")