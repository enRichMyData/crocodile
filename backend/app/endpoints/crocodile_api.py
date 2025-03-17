import os
from typing import Dict, List, Optional

import pandas as pd
from dependencies import get_db
from endpoints.imdb_example import IMDB_EXAMPLE  # Example input
from fastapi import APIRouter, BackgroundTasks, Body, Depends, status
from pydantic import BaseModel
from pymongo.database import Database

router = APIRouter()


class TableUpload(BaseModel):
    table_name: str
    header: List[str]
    total_rows: int
    classified_columns: Optional[Dict[str, Dict[str, str]]] = {}
    data: List[dict]


@router.post("/dataset/{datasetName}/table", status_code=status.HTTP_201_CREATED)
def add_table(
    datasetName: str,
    table_upload: TableUpload = Body(..., example=IMDB_EXAMPLE),
    background_tasks: BackgroundTasks = None,  # <-- Use BackgroundTasks
    db: Database = Depends(get_db),
):
    """
    Add a new table to an existing dataset.
    After onboarding the data, trigger the Crocodile process in the background.
    """
    # Immediately trigger the global Crocodile processing task
    print("Triggering Crocodile task...")

    def run_crocodile_task():
        from crocodile import Crocodile

        croco = Crocodile(
            input_csv=pd.DataFrame(table_upload.data),
            dataset_name=datasetName,
            table_name=table_upload.table_name,
            max_candidates=3,
            entity_retrieval_endpoint=os.environ.get("ENTITY_RETRIEVAL_ENDPOINT"),
            entity_retrieval_token=os.environ.get("ENTITY_RETRIEVAL_TOKEN"),
            max_workers=8,
            candidate_retrieval_limit=10,
            model_path="./crocodile/models/default.h5",
            save_output_to_csv=False,
        )
        croco.run()

    background_tasks.add_task(run_crocodile_task)

    return {
        "message": "Table added successfully.",
        "tableName": table_upload.table_name,
        "datasetName": datasetName,
    }
