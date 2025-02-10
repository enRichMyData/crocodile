# Crocodile

<img src="logo.webp" alt="Crocodile Logo" width="400"/>

**Crocodile** is a powerful Python library designed for efficient entity linking over tabular data. Whether you're working with large datasets or need to resolve entities across multiple tables, Crocodile provides a scalable and easy-to-integrate solution to streamline your data processing pipeline.

> **Fun Fact:** If a crocodile and an alligator were to meet, the crocodile would likely win in a face-to-face combat. While the alligator is faster, the crocodile has the advantage of being bigger, heavier, and having a more lethal bite due to its size and strength ([Bayou Swamp Tours](https://www.bayouswamptours.com/blog/difference-between-alligator-crocodile/)).

## Features

- **Entity Linking:** Seamlessly link entities within tabular data.
- **Scalable:** Designed to handle large datasets efficiently.
- **Easy Integration:** Can be easily integrated into existing data processing pipelines.

## Installation

Install the Crocodile library via pip:

```bash
pip install crocodile
```

## Usage

### 1. Onboarding Data

Before running the entity linking process, onboard your data into MongoDB:

```python
import pandas as pd
from pymongo import MongoClient

# Load the CSV file into a DataFrame
file_path = './imdb_top_1000.csv'
df = pd.read_csv(file_path)

# MongoDB connection
client = MongoClient("mongodb://mongodb:27017/")
db = client["crocodile_db"]
collection = db["input_data"]
trace_collection = db["processing_trace"]

# Dataset and table names for tracing
dataset_name = "imdb_dataset"
table_name = "top_1000_movies"

# Onboard data
for index, row in df.iterrows():
    document = {
        "dataset_name": dataset_name,
        "table_name": table_name,
        "row_id": index,
        "data": row.to_dict(),
        "classified_columns": {
            "NE": ["Series_Title"],  # Assuming Series_Title is the column to be linked
            "LIT": ["Released_Year", "Genre"]  # Assuming these are literal columns
        },
        "context_columns": ["Series_Title", "Released_Year", "Genre", "Director"],  # Context columns
        "status": "TODO"
    }
    collection.insert_one(document)

# Initialize the trace collection
trace_collection.insert_one({
    "dataset_name": dataset_name,
    "table_name": table_name,
    "total_rows": len(df),
    "processed_rows": 0,
    "status": "PENDING"  # Initial status before processing
})

print(f"Data onboarded successfully for dataset '{dataset_name}' and table '{table_name}'.")
```

### 2. Running the Entity Linking Process

Once the data is onboarded, you can run the entity linking process using the `Crocodile` class.

```python
from crocodile import Crocodile
import os

# Create an instance of the Crocodile class
crocodile_instance = Crocodile(
    mongo_uri="mongodb://mongodb:27017/",
    db_name="crocodile_db",
    collection_name="input_data",
    trace_collection_name="processing_trace",
    max_candidates=3,
    entity_retrieval_endpoint=os.environ["ENTITY_RETRIEVAL_ENDPOINT"],  # Access the entity retrieval endpoint directly from environment variables
    entity_retrieval_token=os.environ["ENTITY_RETRIEVAL_TOKEN"]  # Access the entity retrieval token directly from environment variables
)

# Run the entity linking process
crocodile_instance.run(dataset_name=dataset_name, table_name=table_name)

print("Entity linking process completed.")
```

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, feel free to open an issue on the GitHub repository.
