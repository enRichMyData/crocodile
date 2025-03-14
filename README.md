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

You can run the entity linking process using the `Crocodile` class like the following:

```python
from crocodile import Crocodile
import os

file_path = './tables/imdb_top_1000.csv'

# Create an instance of the Crocodile class
crocodile_instance = Crocodile(
    table_name="imdb",
    dataset_name="cinema",
    max_candidates=3,
    entity_retrieval_token=os.environ["ENTITY_RETRIEVAL_TOKEN"]
    entity_retrieval_endpoint=os.environ["ENTITY_RETRIEVAL_ENDPOINT"],
)

# Run the entity linking process
crocodile_instance.run()

print("Entity linking process completed.")
```

If one wants to specify column types for its input table, then it can run the following:

```python
from crocodile import Crocodile
import os

file_path = './tables/imdb_top_1000.csv'

# Create an instance of the Crocodile class
crocodile_instance = Crocodile(
    table_name="imdb",
    dataset_name="cinema",
    max_candidates=3,
    entity_retrieval_token=os.environ["ENTITY_RETRIEVAL_TOKEN"]
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, feel free to open an issue on the GitHub repository.
