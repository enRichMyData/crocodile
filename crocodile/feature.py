from typing import Any, Dict, List, Optional, Set

from crocodile.utils import ngrams, tokenize_text

DEFAULT_FEATURES = [
    "ntoken_mention",
    "ntoken_entity",
    "length_mention",
    "length_entity",
    "popularity",
    "ed_score",
    "jaccard_score",
    "jaccardNgram_score",
    "desc",
    "descNgram",
    "bow_similarity",
    "kind",
    "NERtype",
    "column_NERtype",
    "typeFreq1",
    "typeFreq2",
    "typeFreq3",
    "typeFreq4",
    "typeFreq5",
]


class Feature:
    def __init__(self, features: Optional[List[str]] = None):
        self.selected_features = features or DEFAULT_FEATURES

    def map_kind_to_numeric(self, kind: str) -> int:
        mapping: Dict[str, int] = {
            "entity": 1,
            "type": 2,
            "disambiguation": 3,
            "predicate": 4,
        }
        return mapping.get(kind, 1)

    def map_nertype_to_numeric(self, nertype: str) -> int:
        mapping: Dict[str, int] = {
            "LOCATION": 1,
            "LOC": 1,
            "ORGANIZATION": 2,
            "ORG": 2,
            "PERSON": 3,
            "PERS": 3,
            "OTHER": 4,
            "OTHERS": 4,
        }
        return mapping.get(nertype, 4)

    def calculate_token_overlap(self, tokens_a: Set[str], tokens_b: Set[str]) -> float:
        intersection: Set[str] = tokens_a & tokens_b
        union: Set[str] = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0

    def calculate_ngram_similarity(self, a: str, b: str, n: int = 3) -> float:
        a_ngrams: List[str] = ngrams(a, n)
        b_ngrams: List[str] = ngrams(b, n)
        intersection: int = len(set(a_ngrams) & set(b_ngrams))
        union: int = len(set(a_ngrams) | set(b_ngrams))
        return intersection / union if union > 0 else 0.0

    def process_candidates(
        self, candidates: List[Dict[str, Any]], entity_name: Optional[str], row_tokens: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Process candidate records to calculate a set of features for each candidate.

        Note:
            The fields 'name', 'entity_name', and 'description' might be None.
            For computations we convert them to an empty string, but the output
            preserves the original value.
        """
        # Use a safe version of entity_name for computations.
        safe_entity_name: str = entity_name if entity_name is not None else ""

        processed_candidates: List[Dict[str, Any]] = []
        for candidate in candidates:
            # Retrieve original values, which might be None.
            candidate_name: Optional[str] = candidate.get("name")
            candidate_description: Optional[str] = candidate.get("description")

            # Create safe versions for computation.
            safe_candidate_name: str = candidate_name if candidate_name is not None else ""
            safe_candidate_description: str = (
                candidate_description if candidate_description is not None else ""
            )

            kind_numeric: int = self.map_kind_to_numeric(candidate.get("kind", "entity"))
            nertype_numeric: int = self.map_nertype_to_numeric(candidate.get("NERtype", "OTHERS"))

            desc: float = 0.0
            descNgram: float = 0.0
            if safe_candidate_description:
                desc = self.calculate_token_overlap(
                    row_tokens, tokenize_text(safe_candidate_description)
                )
                descNgram = self.calculate_ngram_similarity(
                    safe_entity_name, safe_candidate_description
                )

            features: Dict[str, Any] = {
                "ntoken_mention": candidate.get("ntoken_mention", len(safe_entity_name.split())),
                "ntoken_entity": candidate.get("ntoken_entity", len(safe_candidate_name.split())),
                "length_mention": len(safe_entity_name),
                "length_entity": len(safe_candidate_name),
                "popularity": candidate.get("popularity", 0.0),
                "ed_score": candidate.get("ed_score", 0.0),
                "jaccard_score": candidate.get("jaccard_score", 0.0),
                "jaccardNgram_score": candidate.get("jaccardNgram_score", 0.0),
                "desc": desc,
                "descNgram": descNgram,
                "bow_similarity": 0.0,
                "kind": kind_numeric,
                "NERtype": nertype_numeric,
                "column_NERtype": None,  # Placeholder it will be filled later
            }

            # Preserve the original candidate values, even if they are None
            processed_candidates.append(
                {
                    "id": candidate.get("id"),
                    "name": candidate_name,
                    "description": candidate_description,
                    "types": candidate.get("types"),
                    "features": features,
                }
            )
        return processed_candidates
