import json
import base64
from typing import Any, Dict, List, Optional, Tuple, Union
from bson import ObjectId
from pymongo.database import Database
from pymongo.collection import Collection

class PaginationService:
    """
    Service for handling pagination in MongoDB and Elasticsearch queries.
    Supports bi-directional keyset pagination using ObjectId or custom fields.
    """
    
    @staticmethod
    def get_mongo_pagination(
        collection: Collection, 
        query_filter: Dict,
        cursor_field: str = "_id",
        next_cursor: Optional[str] = None,
        prev_cursor: Optional[str] = None,
        limit: int = 10
    ) -> Tuple[Dict, int]:
        """
        Setup MongoDB pagination query based on cursors.
        Returns modified filter and sort direction.
        """
        sort_direction = 1  # Default ascending (forward)
        
        if next_cursor and prev_cursor:
            raise ValueError("Only one of next_cursor or prev_cursor should be provided")
            
        if next_cursor:
            # Forward pagination (get items after the cursor)
            try:
                if cursor_field == "_id":
                    query_filter[cursor_field] = {"$gt": ObjectId(next_cursor)}
                else:
                    query_filter[cursor_field] = {"$gt": next_cursor}
            except Exception:
                raise ValueError("Invalid cursor value")
        elif prev_cursor:
            # Backward pagination (get items before the cursor)
            try:
                if cursor_field == "_id":
                    query_filter[cursor_field] = {"$lt": ObjectId(prev_cursor)}
                else:
                    query_filter[cursor_field] = {"$lt": prev_cursor}
                sort_direction = -1  # Sort descending for backward pagination
            except Exception:
                raise ValueError("Invalid cursor value")
                
        return query_filter, sort_direction

    @staticmethod
    def process_mongo_results(
        results: List[Dict],
        limit: int,
        collection: Collection,
        query_filter: Dict,
        cursor_field: str = "_id",
        sort_direction: int = 1,
        id_field: str = "_id"
    ) -> Tuple[List[Dict], Optional[str], Optional[str]]:
        """
        Process MongoDB query results and calculate next/prev cursors.
        """
        # Check if we have more items than requested
        has_more = len(results) > limit
        if has_more:
            results = results[:limit]  # Remove the extra item
            
        # If we did backward pagination, reverse the results
        if sort_direction == -1:
            results.reverse()
            
        # Get cursors for next and previous pages
        next_cursor = None
        prev_cursor = None
        
        if results:
            # Calculate prev_cursor
            if not query_filter.get(cursor_field, {}).get("$gt"):  # No forward pagination filter
                if sort_direction == -1:  # We came backwards
                    prev_cursor = str(results[0][id_field])
                else:
                    # Check if there are documents before the current page
                    first_id = results[0][id_field]
                    prev_filter = {k: v for k, v in query_filter.items() if k != cursor_field}
                    
                    if cursor_field == "_id":
                        prev_filter[cursor_field] = {"$lt": first_id}
                    else:
                        prev_filter[cursor_field] = {"$lt": first_id}
                        
                    if collection.count_documents(prev_filter) > 0:
                        prev_cursor = str(first_id)
            else:
                # We came from a next_cursor, there are previous items
                prev_cursor = str(results[0][id_field])
                
            # Calculate next_cursor
            if has_more:
                next_cursor = str(results[-1][id_field])
            elif sort_direction == -1:  # We came backwards
                next_cursor = str(results[-1][id_field])
                
        return results, next_cursor, prev_cursor

    @staticmethod
    def create_es_cursor(sort_values: List[Any]) -> str:
        """Create an Elasticsearch cursor from sort values"""
        cursor_data = {"sort": sort_values}
        return base64.b64encode(json.dumps(cursor_data).encode('utf-8')).decode('utf-8')

    @staticmethod
    def parse_es_cursor(cursor: str) -> List[Any]:
        """Parse an Elasticsearch cursor to extract sort values"""
        try:
            cursor_data = json.loads(base64.b64decode(cursor).decode('utf-8'))
            return cursor_data.get("sort", [])
        except (json.JSONDecodeError, ValueError, KeyError):
            raise ValueError("Invalid cursor format")
