# Crocodile

<img src="logo.webp" alt="Crocodile Logo" width="400"/>

**Crocodile** is a powerful Python library designed for efficient entity linking over tabular data. Whether you're working with large datasets or need to resolve entities across multiple tables, Crocodile provides a scalable and easy-to-integrate solution to streamline your data processing pipeline.

> **Fun Fact:** If a crocodile and an alligator were to meet, the crocodile would likely win in a face-to-face combat. While the alligator is faster, the crocodile has the advantage of being bigger, heavier, and having a more lethal bite due to its size and strength ([Bayou Swamp Tours](https://www.bayouswamptours.com/blog/difference-between-alligator-crocodile/)).

## Features

- **Entity Linking:** Seamlessly link entities within tabular data.
- **Scalable:** Designed to handle large datasets efficiently.
- **Easy Integration:** Can be easily integrated into existing data processing pipelines.

## Installation

Crocodile is not yet available on PyPI. To install it, clone the repository and install it manually:

```bash
git clone https://github.com/your-org/crocodile.git
cd crocodile
pip install -e .
```

Additionally, one needs to download the SpaCy model by running the following code:

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Using the CLI
You can run the entity linking process via the command line interface (CLI) as follows:

First, create a `.env` file with the required environment variables:

```ini
ENTITY_RETRIEVAL_ENDPOINT=https://lamapi.hel.sintef.cloud/lookup/entity-retrieval
ENTITY_RETRIEVAL_TOKEN=lamapi_demo_2023
```

Then, use the following command:

```bash
python3 -m crocodile.cli \
  --croco.input_csv tables/imdb_top_1000.csv \
  --croco.entity_retrieval_endpoint "$ENTITY_RETRIEVAL_ENDPOINT" \
  --croco.entity_retrieval_token "$ENTITY_RETRIEVAL_TOKEN" \
  --croco.mongo_uri "localhost:27017"
```

#### Specifying Column Types via CLI
To specify column types for your input table, use the following command:

```bash
python3 -m crocodile.cli \
  --croco.input_csv tables/imdb_top_1000.csv \
  --croco.entity_retrieval_endpoint "$ENTITY_RETRIEVAL_ENDPOINT" \
  --croco.entity_retrieval_token "$ENTITY_RETRIEVAL_TOKEN" \
  --croco.columns_type '{
    "NE": { "0": "OTHER" },
    "LIT": {
      "1": "NUMBER",
      "2": "NUMBER",
      "3": "STRING",
      "4": "NUMBER",
      "5": "STRING"
    },
    "IGNORED": ["6", "9", "10", "7", "8"]
  }' \
  --croco.mongo_uri "localhost:27017"
```

### Using Python API
You can also run the entity linking process using the `Crocodile` class in Python:

```python
from crocodile import Crocodile
import pandas as pd
import os

df = pd.read_csv("./tables/imdb_top_1000.csv")

croco = Crocodile(
    input_csv=df, 
    dataset_name="cinema",
    table_name="imdb",
    entity_retrieval_endpoint=os.environ["ENTITY_RETRIEVAL_ENDPOINT"],
    entity_retrieval_token=os.environ["ENTITY_RETRIEVAL_TOKEN"],
    candidate_retrieval_limit=10,
    max_workers=4,
    save_output_to_csv=False,
    return_dataframe=True         
)

result_df = croco.run()
print("Entity linking completed.")
```

### Specifying Column Types
If you want to specify column types for your input table, use the following example:

```python
from crocodile import Crocodile
import os

file_path = './tables/imdb_top_1000.csv'

# Create an instance of the Crocodile class
crocodile_instance = Crocodile(
    table_name="imdb",
    dataset_name="cinema",
    max_candidates=3,
    entity_retrieval_token=os.environ["ENTITY_RETRIEVAL_TOKEN"],
    entity_retrieval_endpoint=os.environ["ENTITY_RETRIEVAL_ENDPOINT"],
    columns_type={
        "NE": {
            "0": "OTHER"
        },
        "LIT": {
            "1": "NUMBER",
            "2": "NUMBER",
            "3": "STRING",
            "4": "NUMBER",
            "5": "STRING"
        },
        "IGNORED" : ["6", "9", "10", "7", "8"]
    }
)

# Run the entity linking process
crocodile_instance.run()

print("Entity linking process completed.")
```

In the `columns_type` parameter, one has to specify **for every column index** whether it is a Named-Entity (NE) column or a Literal (LIT) one. All the columns that are not specified neither as NE nor as LIT will be considered as IGNORED columns.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, feel free to open an issue on the GitHub repository.
