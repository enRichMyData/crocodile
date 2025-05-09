from typing import Dict, List, Optional
from pydantic import BaseModel, Field

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

class TableUpload(BaseModel):
    """Model for JSON table upload"""
    table_name: str
    header: List[str]
    total_rows: int
    classified_columns: Optional[Dict[str, Dict[str, str]]] = Field(default_factory=dict)
    data: List[dict]
    
class DatasetCreate(BaseModel):
    """Model for dataset creation"""
    dataset_name: str
