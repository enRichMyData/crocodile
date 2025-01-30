from fastapi import APIRouter, Depends
from pymongo.database import Database
from dependencies import get_db

router = APIRouter()

@router.get("/db_name")
def example_endpoint(db: Database = Depends(get_db)):
    return {"database": db.name}