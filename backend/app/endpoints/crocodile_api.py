from fastapi import APIRouter

router = APIRouter()

@router.get("/crocodile")
def get_crocodile_info():
    return {"crocodile": "Crocodile"}