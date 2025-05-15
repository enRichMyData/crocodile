from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator # type: ignore


# Common Schemas
class DeleteResponse(BaseModel):
    """Response model for delete operations."""
    
    message: str


class Pagination(BaseModel):
    """Pagination information for API responses."""

    next_cursor: Optional[str] = None
    prev_cursor: Optional[str] = None


# Dataset Schemas
class DatasetBase(BaseModel):
    """Base model for dataset information."""

    dataset_name: str = Field(..., description="Name of the dataset.")
    total_tables: int = Field(default=0, description="Total number of tables in the dataset.")
    total_rows: int = Field(default=0, description="Total number of rows across all tables in the dataset.")
    user_id: str = Field(..., description="Identifier of the user who owns the dataset.")
    created_at: str = Field(..., description="Timestamp of when the dataset was created (ISO format).")


class DatasetResponseItem(DatasetBase):
    """Response model for a dataset item."""

    id: str = Field(..., alias="_id", description="Unique identifier of the dataset.")

    class Config:
        populate_by_name = True # Allows using alias "_id" for "id"


class DatasetCreateResponse(BaseModel):
    """Response model for creating a new dataset."""

    message: str
    dataset: DatasetResponseItem


class DatasetListResponse(BaseModel):
    """Response model for listing datasets."""

    data: List[DatasetResponseItem]
    pagination: Pagination


class DatasetCreateRequest(BaseModel):
    """Request model for creating a new dataset."""

    dataset_name: str = Field(..., min_length=1, description="Name of the dataset, cannot be empty.")
    
    @model_validator(mode='after')
    def check_dataset_name_not_empty(cls, values):
        dataset_name = values.dataset_name
        if dataset_name.strip() == '':
            raise ValueError('Dataset name cannot contain only whitespace')
        return values


# Table Schemas
class TableUpload(BaseModel):
    """Request model for validating table uploads"""

    table_name: str = Field(..., min_length=1, description="Name of the table, cannot be empty.")
    header: List[str] = Field(..., min_items=1, description="List of column headers, cannot be empty.")
    total_rows: int = Field(..., ge=0, description="Total number of rows in the data, cannot be negative.")
    classified_columns: Optional[Dict[str, Dict[str, str]]] = Field(default_factory=dict)
    data: List[dict]

    @model_validator(mode='after')
    def check_table_name_not_empty(cls, values):
        table_name = values.table_name
        if table_name.strip() == '':
            raise ValueError('Table name cannot contain only whitespace')
        return values

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


class CSVTableUpload(BaseModel):
    """Request model for validating CSV table uploads"""

    table_name: str = Field(..., min_length=1, description="Name of the table, cannot be empty.")
    
    @model_validator(mode='after')
    def check_table_name_not_empty(cls, values):
        table_name = values.table_name
        if table_name.strip() == '':
            raise ValueError('Table name cannot contain only whitespace')
        return values
    
class TableAddResponse(BaseModel):
    """Response model for adding a table to a dataset."""

    message: str
    tableName: str
    datasetName: str
    userId: str


# Annotation Schemas
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


# Table Response Schemas
class TableResponseItem(BaseModel):
    """Response model for a table item."""
    
    id: str = Field(..., alias="_id", description="Unique identifier of the table.")
    table_name: str = Field(..., description="Name of the table.")
    dataset_name: str = Field(..., description="Name of the dataset the table belongs to.")
    user_id: str = Field(..., description="Identifier of the user who owns the table.")
    total_rows: int = Field(default=0, description="Total number of rows in the table.")
    header: List[str] = Field(default_factory=list, description="Column headers for the table.")
    created_at: str = Field(..., description="Timestamp of when the table was created (ISO format).")
    completed_at: Optional[str] = Field(None, description="Timestamp of when processing was completed (ISO format).")

    class Config:
        populate_by_name = True # Allows using alias "_id" for "id"


class TableListResponse(BaseModel):
    """Response model for listing tables in a dataset."""
    
    dataset: str
    data: List[TableResponseItem]
    pagination: Pagination


class LinkedEntity(BaseModel):
    """Model for linked entity information in a table row."""
    
    idColumn: int
    candidates: List[Dict[str, Any]]


class TableRowItem(BaseModel):
    """Model for a row in a table with linked entities."""
    
    idRow: int
    data: List[Any]
    linked_entities: List[LinkedEntity] = Field(default_factory=list)


class TableData(BaseModel):
    """Model for table data with rows and metadata."""
    
    datasetName: str
    tableName: str
    status: str
    header: List[str]
    rows: List[TableRowItem]


class TableRowsResponse(BaseModel):
    """Response model for getting rows from a specific table."""
    
    data: TableData
    pagination: Pagination


class AnnotationUpdate(BaseModel):
    """Request model for updating an annotation."""

    entity_id: str
    match: bool = True  # Whether this is the correct entity
    score: Optional[float] = 1.0  # Default to 1.0 for user selections
    notes: Optional[str] = None
    # If providing a new candidate not in the existing list
    candidate_info: Optional[EntityCandidate] = None
