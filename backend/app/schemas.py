from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Models for column metadata
class ColumnMetadata(BaseModel):
    idColumn: int
    tag: str
    datatype: Optional[str] = None

# Models for semantic annotations
class SemanticAnnotations(BaseModel):
    cea: List = []
    cta: List = []
    cpa: List = []

# Models for table metadata
class TableMetadata(BaseModel):
    column: List[ColumnMetadata]

# Model for row data
class TableRow(BaseModel):
    idRow: int
    data: List[str]

# Model for adding a table to a dataset
class TableItem(BaseModel):
    tableName: str
    header: List[str]
    rows: List[TableRow]
    semanticAnnotations: Optional[SemanticAnnotations] = Field(default_factory=SemanticAnnotations)
    metadata: Optional[TableMetadata] = None
    kgReference: str = "wikidata"

# Model for adding a dataset with a table
class DatasetItem(BaseModel):
    datasetName: str
    tableName: str
    header: List[str]
    rows: List[TableRow]
    semanticAnnotations: Optional[SemanticAnnotations] = Field(default_factory=SemanticAnnotations)
    metadata: Optional[TableMetadata] = None
    kgReference: str = "wikidata"

# Models for entity linking results
class EntityInfo(BaseModel):
    id: str
    name: str
    description: str

class LinkedEntity(BaseModel):
    idColumn: int
    entity: Optional[EntityInfo] = None

# Models for pagination
class PaginationInfo(BaseModel):
    nextCursor: Optional[str] = None
    previousCursor: Optional[str] = None

# Response models for different endpoints
class DatasetListItem(BaseModel):
    datasetName: str
    tablesCount: int
    status: str

class DatasetList(BaseModel):
    datasets: List[DatasetListItem]
    pagination: PaginationInfo

class TableListItem(BaseModel):
    tableName: str
    status: str

class TableList(BaseModel):
    tables: List[TableListItem]
    pagination: PaginationInfo

class ProcessedRow(BaseModel):
    idRow: int
    data: List[str]
    linked_entities: List[LinkedEntity]

class TableData(BaseModel):
    datasetName: str
    tableName: str
    header: List[str]
    rows: List[ProcessedRow]

class TableResults(BaseModel):
    data: TableData
    pagination: PaginationInfo