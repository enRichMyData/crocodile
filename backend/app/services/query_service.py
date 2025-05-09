from elasticsearch import Elasticsearch
from typing import Dict, List, Optional, Any, Union
from services.pagination_service import PaginationService

class QueryService:
    """
    Service for constructing and executing complex queries in Elasticsearch and MongoDB.
    """
    
    @staticmethod
    def build_es_table_query(
        user_id: str,
        dataset_name: str,
        table_name: str,
        search: Optional[str] = None,
        search_columns: Optional[List[int]] = None,
        column: Optional[int] = None,
        include_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        sort_direction: str = "desc",
        search_after: Optional[List[Any]] = None,
        search_before: Optional[List[Any]] = None,
        limit: int = 10,
        es_index: str = "table_rows"
    ) -> Dict:
        """
        Build an Elasticsearch query for table data with all filters and sorting options.
        """
        # Determine if we're doing backward pagination
        is_backward = search_before is not None
        
        # Build base query filters
        filters = [
            {"term": {"user_id": user_id}},
            {"term": {"dataset_name": dataset_name}},
            {"term": {"table_name": table_name}},
        ]
        
        # Add text search filter if provided
        if search:
            if search_columns:
                # Search in specified columns only (OR condition across columns)
                nested_queries = []
                for col_idx in search_columns:
                    nested_queries.append({
                        "bool": {
                            "must": [
                                {"match": {"data.value": search}},
                                {"term": {"data.col_index": col_idx}}
                            ]
                        }
                    })
                
                filters.append({
                    "nested": {
                        "path": "data",
                        "query": {"bool": {"should": nested_queries, "minimum_should_match": 1}}
                    }
                })
            elif column is not None:
                # Search in specified column
                filters.append({
                    "nested": {
                        "path": "data",
                        "query": {
                            "bool": {
                                "must": [
                                    {"match": {"data.value": search}},
                                    {"term": {"data.col_index": column}}
                                ]
                            }
                        }
                    }
                })
            else:
                # Search in all columns
                filters.append({
                    "nested": {
                        "path": "data",
                        "query": {"bool": {"must": [{"match": {"data.value": search}}]}}
                    }
                })
        
        # Add type filters if provided - must specify a column
        if (include_types or exclude_types):
            type_column = column  # Use the unified column parameter
            
            if type_column is None:
                raise ValueError("Must specify 'column' parameter when filtering by types")
                
            # Include specific types (ANY of the specified types must match)
            if include_types:
                type_filter_query = {
                    "nested": {
                        "path": "data",
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"data.col_index": type_column}},
                                    {"bool": {"should": [{"term": {"data.types": type_name}} for type_name in include_types]}}
                                ]
                            }
                        }
                    }
                }
                filters.append(type_filter_query)
            
            # Exclude specific types (NONE of the specified types should match)
            if exclude_types:
                for type_name in exclude_types:
                    exclude_filter = {
                        "bool": {
                            "must_not": {
                                "nested": {
                                    "path": "data",
                                    "query": {
                                        "bool": {
                                            "must": [
                                                {"term": {"data.col_index": type_column}},
                                                {"term": {"data.types": type_name}}
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    }
                    filters.append(exclude_filter)
        
        # Build sort criteria
        sort_criteria = []
        
        # Set up sorting based on confidence if requested
        if sort_by == "confidence":
            # Column-level confidence sort - requires column parameter
            if column is None:
                raise ValueError("Must specify 'column' parameter when sorting by column confidence")
                
            # Use the working format for nested sort
            sort_criteria.append({
                "data.confidence": {
                    "order": "asc" if is_backward else sort_direction,
                    "nested": {
                        "path": "data",
                        "filter": {"term": {"data.col_index": column}}
                    },
                    "missing": "_last" if sort_direction == "desc" else "_first"
                }
            })
        elif sort_by == "confidence_avg":
            # Row-level average confidence sort
            sort_criteria.append({
                "avg_confidence": {
                    "order": "asc" if is_backward else sort_direction,
                    "missing": "_last" if sort_direction == "desc" else "_first"
                }
            })
        
        # Always add row_id as secondary sort for stable pagination
        sort_criteria.append({
            "row_id": {
                "order": "asc" if is_backward else "asc"  # always ascending for row_id
            }
        })
            
        # Prepare the search body
        body = {
            "query": {"bool": {"filter": filters}},
            "sort": sort_criteria,
            "_source": ["row_id", "avg_confidence"],  # Include avg_confidence in results
            "size": limit + 1,      # Get one extra to check for more results
            "track_total_hits": True  # Get accurate total hit count
        }
        
        # Add search_after for forward pagination
        if search_after:
            body["search_after"] = search_after
        
        # For backward pagination, we need to reverse the sort order temporarily
        # and then reverse the results afterward
        if is_backward and search_before:
            # Reverse sort orders for backward pagination
            for sort_item in body["sort"]:
                for key in sort_item:
                    if isinstance(sort_item[key], dict) and "order" in sort_item[key]:
                        sort_item[key]["order"] = "desc" if sort_item[key]["order"] == "asc" else "asc"
            
            body["search_after"] = search_before
            
        return body
