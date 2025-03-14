from typing import Dict, List, Literal, TypedDict

LitType = Literal["NUMBER", "STRING", "DATETIME"]
NerType = Literal["LOCATION", "ORGANIZATION", "PERSON", "OTHER"]


class ColType(TypedDict):
    NE: Dict[str, LitType | NerType]
    LIT: Dict[str, LitType | NerType]
    IGNORED: List[str]
