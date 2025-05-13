from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator # type: ignore


class TableUpload(BaseModel):
    table_name: str = Field(..., min_length=1, description="Name of the table, cannot be empty.")
    header: List[str] = Field(..., min_items=1, description="List of column headers, cannot be empty.")
    total_rows: int = Field(..., ge=0, description="Total number of rows in the data, cannot be negative.")
    classified_columns: Optional[Dict[str, Dict[str, str]]] = Field(default_factory=dict)
    data: List[dict]

    @model_validator(mode='after')
    def check_data_consistency(cls, values):
        # Check if 'data' and 'total_rows' attributes exist, which they should if validation passed so far.
        data = getattr(values, 'data', None)
        total_rows = getattr(values, 'total_rows', None)

        # This check should only run if both fields are present and correctly typed.
        if data is not None and total_rows is not None:
            if len(data) != total_rows:
                raise ValueError(
                    f"The number of rows in 'data' ({len(data)}) must match 'total_rows' ({total_rows})."
                )
        return values


class EntityType(BaseModel):
    """Type information for an entity"""

    id: str
    name: str


class EntityCandidate(BaseModel):
    """Complete entity candidate information without matching status"""

    id: str
    name: str
    description: str
    types: List[EntityType]
    # Note: score and match are handled at the annotation level


class AnnotationUpdate(BaseModel):
    """Request model for updating an annotation."""

    entity_id: str
    match: bool = True  # Whether this is the correct entity
    score: Optional[float] = 1.0  # Default to 1.0 for user selections
    notes: Optional[str] = None
    # If providing a new candidate not in the existing list
    candidate_info: Optional[EntityCandidate] = None


class TableAddResponse(BaseModel):
    message: str
    tableName: str
    datasetName: str
    userId: str


# Dataset Schemas
class DatasetBase(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset.")
    total_tables: int = Field(default=0, description="Total number of tables in the dataset.")
    total_rows: int = Field(default=0, description="Total number of rows across all tables in the dataset.")
    user_id: str = Field(..., description="Identifier of the user who owns the dataset.")
    created_at: str = Field(..., description="Timestamp of when the dataset was created (ISO format).")


class DatasetResponseItem(DatasetBase):
    id: str = Field(..., alias="_id", description="Unique identifier of the dataset.")

    class Config:
        populate_by_name = True # Allows using alias "_id" for "id"


class DatasetCreateResponse(BaseModel):
    message: str
    dataset: DatasetResponseItem


class Pagination(BaseModel):
    next_cursor: Optional[str] = None
    prev_cursor: Optional[str] = None


class DatasetListResponse(BaseModel):
    data: List[DatasetResponseItem]
    pagination: Pagination


class DatasetCreateRequest(BaseModel):
    dataset_name: str = Field(..., min_length=1, description="Name of the dataset, cannot be empty.")


class DatasetDeleteResponse(BaseModel):
    message: str
