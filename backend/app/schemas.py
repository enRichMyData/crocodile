from typing import List, Optional
from pydantic import BaseModel

class TableMetadata(BaseModel):
    idColumn: int
    tag: str
    datatype: Optional[str] = None

class DatasetMetadata(BaseModel):
    column: List[TableMetadata]

class TableItem(BaseModel):
    tableName: str
    header: List[str]
    rows: List[List[str]]  # 
    semanticAnnotations: dict  
    metadata: DatasetMetadata  
    kgReference: str  

class DatasetItem(BaseModel):
    datasetName: str
    tables: List[TableItem]  

