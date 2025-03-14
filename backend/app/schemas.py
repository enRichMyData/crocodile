from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# Models for Candidate Entity Linking (EL) Results
class ELResultType(BaseModel):
    id: str
    name: str


class ELResultFeature(BaseModel):
    ntoken_mention: int
    ntoken_entity: int
    length_mention: int
    length_entity: int
    popularity: float
    ed_score: float
    jaccard_score: float
    jaccardNgram_score: float
    desc: float
    descNgram: float
    bow_similarity: float
    kind: int
    NERtype: int
    column_NERtype: int


class ELResultCandidate(BaseModel):
    id: str
    name: str
    description: str
    types: List[ELResultType]
    features: ELResultFeature
    score: float


# Model for Classified Columns (NE, LIT, UNCLASSIFIED)
class ClassifiedColumns(BaseModel):
    NE: Optional[Dict[str, str]] = {}
    LIT: Optional[Dict[str, str]] = {}
    UNCLASSIFIED: Optional[Dict[str, str]] = {}


# Model for a Row in the input_data collection
class RowItem(BaseModel):
    dataset_name: str
    table_name: str
    row_id: int
    # Data stored as a list of values (order follows the header)
    data: List[Any]
    classified_columns: ClassifiedColumns
    # List of column indices (as strings)
    context_columns: List[str]
    # Ground-truth QIDs; initially empty
    correct_qids: Dict[str, Any] = {}
    # Processing status (e.g. "TODO", "DONE")
    status: str
    # Optional entity linking results; keys correspond to column indices
    el_results: Optional[Dict[str, List[ELResultCandidate]]] = None


# Model for table-level metadata in the table_trace collection
class TableTrace(BaseModel):
    dataset_name: str
    table_name: str
    header: List[str]
    total_rows: int
    processed_rows: int
    status: str
    start_time: datetime
    completion_percentage: float
    rows_per_second: float
    status_counts: Dict[str, int]
    time_passed_seconds: float
    end_time: datetime


# Model for dataset-level metadata in the dataset_trace collection
class DatasetTrace(BaseModel):
    dataset_name: str
    processed_rows: int
    processed_tables: int
    status: str
    total_rows: int
    total_tables: int
    start_time: datetime
    completion_percentage: float
    rows_per_second: float
    status_counts: Dict[str, int]
    time_passed_seconds: float
    end_time: datetime
