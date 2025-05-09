from typing import Dict, List, Optional, Any, Tuple
from pymongo.database import Database
from endpoints.models import AnnotationUpdate

class AnnotationService:
    """
    Service for annotation-related operations including updating and deleting annotations.
    """
    
    @staticmethod
    def update_candidate_list(
        candidates: List[Dict],
        annotation: AnnotationUpdate
    ) -> Tuple[List[Dict], Dict]:
        """
        Update the candidate list based on an annotation update request.
        Returns the updated candidates list and the matched candidate.
        """
        # Check if the entity exists in the candidates
        entity_found = any(candidate.get("id") == annotation.entity_id for candidate in candidates)
        updated_candidates = []
        matched_candidate = None

        if not entity_found:
            if not annotation.candidate_info:
                raise ValueError(
                    f"Entity with ID {annotation.entity_id} not found in candidates. "
                    "Please provide 'candidate_info' to add a new candidate."
                )

            # Convert Pydantic model to dict for MongoDB storage and add annotation data
            new_candidate = annotation.candidate_info.dict()
            new_candidate["match"] = annotation.match
            new_candidate["score"] = annotation.score if annotation.match else None
            if annotation.notes:
                new_candidate["notes"] = annotation.notes
                
            matched_candidate = new_candidate

            # Add all other candidates with match=False and score=null
            for candidate in candidates:
                # Skip if we already have this id (prevent duplicates)
                if candidate.get("id") == annotation.entity_id:
                    continue

                candidate_copy = dict(candidate)
                candidate_copy["match"] = False
                candidate_copy["score"] = None
                updated_candidates.append(candidate_copy)

            # Add the new candidate
            updated_candidates.append(new_candidate)

        else:
            # Update existing candidates
            for candidate in candidates:
                # Skip if we already have this id (prevent duplicates)
                if candidate.get("id") in [c.get("id") for c in updated_candidates]:
                    continue

                candidate_copy = dict(candidate)

                # If this is the target entity, update it
                if candidate_copy.get("id") == annotation.entity_id:
                    candidate_copy["match"] = annotation.match
                    candidate_copy["score"] = annotation.score if annotation.match else None
                    if annotation.notes:
                        candidate_copy["notes"] = annotation.notes
                    matched_candidate = candidate_copy
                else:
                    # Ensure other candidates are not matched
                    candidate_copy["match"] = False
                    candidate_copy["score"] = None

                updated_candidates.append(candidate_copy)

        # Sort candidates - matched candidate first
        updated_candidates.sort(key=lambda x: (0 if x.get("match") else 1))
        
        return updated_candidates, matched_candidate
    
    @staticmethod
    def delete_candidate(
        candidates: List[Dict],
        entity_id: str
    ) -> List[Dict]:
        """
        Delete a candidate from a list and update matching status if needed.
        """
        # Check if the entity exists in the candidates
        entity_exists = any(candidate.get("id") == entity_id for candidate in candidates)
        
        if not entity_exists:
            raise ValueError(f"Entity with ID {entity_id} not found in candidates")

        # Create updated candidates list without the specified entity
        updated_candidates = [c for c in candidates if c.get("id") != entity_id]

        # Check if we're removing a matched candidate, reorder if needed
        was_matched = any(c.get("id") == entity_id and c.get("match", False) for c in candidates)
        
        if was_matched and updated_candidates:
            # The deleted candidate was the matched one, select the first remaining one
            updated_candidates[0]["match"] = True
            updated_candidates[0]["score"] = 1.0  # Default score for manually selected
            
        return updated_candidates
