from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from column_classifier import ColumnClassifier
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError
from services.utils import format_classification, log_error, log_info


class DataService:
    """
    Service for handling data operations including:
    - Table creation and management
    - Row insertion
    - Data formatting and classification
    """

    @staticmethod
    def create_or_get_dataset(db: Database, user_id: str, dataset_name: str) -> Dict[str, Any]:
        """
        Create a new dataset or get an existing one.

        Args:
            db: MongoDB database
            user_id: User ID
            dataset_name: Dataset name

        Returns:
            Dictionary with dataset information and ID

        Raises:
            ValueError: If there was an error creating the dataset
        """
        # Check if dataset exists
        dataset = db.datasets.find_one({"user_id": user_id, "dataset_name": dataset_name})

        if not dataset:
            try:
                # Create new dataset
                dataset_data = {
                    "user_id": user_id,
                    "dataset_name": dataset_name,
                    "created_at": datetime.now(),
                    "total_tables": 0,
                    "total_rows": 0,
                }

                dataset_id = db.datasets.insert_one(dataset_data).inserted_id
                dataset_data["_id"] = dataset_id
                return dataset_data

            except DuplicateKeyError:
                raise ValueError("Duplicate dataset insertion")

        return dataset

    @staticmethod
    def get_or_create_column_classification(
        data: pd.DataFrame, header: List[str], provided_classification: Optional[Dict] = None
    ) -> Dict:
        """
        Get or create column classification for a table.

        Args:
            data: DataFrame containing the data
            header: List of column headers
            provided_classification: Optional pre-defined classification

        Returns:
            Dictionary with column classification
        """
        if provided_classification:
            return provided_classification

        # Auto-classify columns
        classifier = ColumnClassifier(model_type="fast")
        classification_result = classifier.classify_multiple_tables([data.head(1024)])
        raw_classification = classification_result[0].get("table_1", {})
        return format_classification(raw_classification, header)

    @staticmethod
    def create_table(
        db: Database,
        user_id: str,
        dataset_name: str,
        table_name: str,
        header: List[str],
        total_rows: int,
        classification: Dict,
        data_df: Optional[pd.DataFrame] = None,
        data_list: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Create a new table in the dataset.

        Args:
            db: MongoDB database
            user_id: User ID
            dataset_name: Dataset name
            table_name: Table name
            header: List of column headers
            total_rows: Total number of rows
            classification: Column classification dictionary
            data_df: Optional DataFrame containing the data
            data_list: Optional list containing the data

        Returns:
            Dictionary with table information

        Raises:
            ValueError: If there was an error creating the table
        """
        # Get or create dataset
        dataset = DataService.create_or_get_dataset(db, user_id, dataset_name)
        dataset_id = dataset.get("_id")

        # Create table metadata
        table_metadata = {
            "user_id": user_id,
            "dataset_name": dataset_name,
            "table_name": table_name,
            "header": header,
            "total_rows": total_rows,
            "created_at": datetime.now(),
            "status": "DOING",
            "classified_columns": classification,
        }

        try:
            db.tables.insert_one(table_metadata)
        except DuplicateKeyError:
            raise ValueError(f"Table with name '{table_name}' already exists in dataset")

        # Update dataset metadata
        db.datasets.update_one(
            {"_id": dataset_id}, {"$inc": {"total_tables": 1, "total_rows": total_rows}}
        )

        # Insert rows into the database
        input_data = []

        if data_df is not None:
            # Process DataFrame
            for i, (_, row) in enumerate(data_df.iterrows()):
                row_values = row.replace({np.nan: None}).tolist()

                input_doc = {
                    "user_id": user_id,
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "row_id": i,
                    "data": row_values,
                    "status": "TODO",
                    "el_results": {},
                    "ml_status": "TODO",
                    "manually_annotated": False,
                    "created_at": datetime.now(),
                }
                input_data.append(input_doc)

        elif data_list is not None:
            # Process list of dictionaries or lists
            for i, row_data in enumerate(data_list):
                # Convert row data to list format if it's a dict
                if isinstance(row_data, dict):
                    row_values = [row_data.get(col, None) for col in header]
                else:
                    row_values = row_data

                input_doc = {
                    "user_id": user_id,
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "row_id": i,
                    "data": row_values,
                    "status": "TODO",
                    "el_results": {},
                    "ml_status": "TODO",
                    "manually_annotated": False,
                    "created_at": datetime.now(),
                }
                input_data.append(input_doc)

        # Store rows in database
        if input_data:
            try:
                # Insert into MongoDB
                db.input_data.insert_many(input_data)
                log_info(f"Stored {len(input_data)} rows in database for {dataset_name}/{table_name}")
                
            except Exception as e:
                log_error(f"Error storing rows in database: {str(e)}", e)
                # We'll continue anyway since Crocodile will handle processing

        return table_metadata
