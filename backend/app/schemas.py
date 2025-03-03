from typing import List, Optional
from pydantic import BaseModel

class TableMetadata(BaseModel):
    idColumn: int
    tag: str
    datatype: Optional[str] = None

class DatasetMetadata(BaseModel):
    column: List[TableMetadata]

class DatasetItem(BaseModel):
    datasetName: str
    tableName: str
    header: List[str]
    rows: List[dict]
    semanticAnnotations: dict
    metadata: DatasetMetadata
    kgReference: str

class TableItem(BaseModel):
    tableName: str
    header: List[str]
    rows: List[dict]
    semanticAnnotations: dict
    metadata: DatasetMetadata
    kgReference: str
