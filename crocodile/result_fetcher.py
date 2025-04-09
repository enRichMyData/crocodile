from typing import Dict, List, Optional, Union

from crocodile.mongo import MongoConnectionManager, MongoWrapper


class CrocodileResultFetcher:
    """
    Dedicated class for fetching entity linking results from MongoDB.
    
    This class handles retrieving results in various formats:
    - Raw results by row_ids
    - Custom projections and filters
    - Status counts and statistics
    """
    
    def __init__(
        self,
        client_id: str,
        dataset_name: str,
        table_name: str,
        **kwargs
    ):
        """
        Initialize the CrocodileResultFetcher with dataset identification parameters.
        
        Args:
            client_id: The client identifier
            dataset_name: The dataset name
            table_name: The table name
            **kwargs: Additional parameters including mongo_uri, db_name, etc.
        """
        self.client_id = client_id
        self.dataset_name = dataset_name  
        self.table_name = table_name
        self._db_name = kwargs.get("db_name", "crocodile_db")
        self._mongo_uri = kwargs.get("mongo_uri", "mongodb://mongodb:27017/")
        self.input_collection = kwargs.get("input_collection", "input_data")
        self.mongo_wrapper = MongoWrapper(self._mongo_uri, self._db_name)
    
    def get_db(self):
        """Get MongoDB database connection for current process"""
        client = MongoConnectionManager.get_client(self._mongo_uri)
        return client[self._db_name]
    
    def get_results(self, row_ids: Optional[Union[int, List[int]]] = None) -> List[Dict]:
        """
        Read entity linking results by specific row_ids.
        
        This method efficiently retrieves raw entity linking results for specific row IDs.
        
        Args:
            row_ids: List of specific row IDs to retrieve or a single row_id.
                    If None, raises an error.
            
        Returns:
            A list of dictionaries containing the raw entity linking results for the specified row_ids,
            along with row_id and status fields for tracking processing state.
        
        Raises:
            ValueError: If row_ids is None or empty
        """
        if row_ids is None or (isinstance(row_ids, list) and len(row_ids) == 0):
            raise ValueError("row_ids parameter must be specified with one or more row IDs")
            
        db = self.get_db()
        input_collection = db[self.input_collection]
        
        # Normalize row_ids to list format
        if not isinstance(row_ids, list):
            row_ids = [row_ids]
            
        # Construct the query
        query = {
            "client_id": self.client_id,
            "dataset_name": self.dataset_name,
            "table_name": self.table_name,
            "row_id": {"$in": row_ids}
        }
        
        # Setup projection to only fetch fields we need
        projection = {
            "el_results": 1, 
            "row_id": 1,
            "status": 1,
            "ml_status": 1
        }
        
        # Query directly for the specified row_ids
        cursor = input_collection.find(query, projection=projection).sort("row_id", 1)
        
        # Process results - just grab the raw el_results and metadata
        results = []
        for doc in cursor:
            result = {
                "row_id": doc["row_id"],
                "status": doc.get("status", "UNKNOWN"),
                "ml_status": doc.get("ml_status", "UNKNOWN"),
                "el_results": doc.get("el_results", {})
            }
            results.append(result)
        
        # Return results directly as a list
        return results
    
    def get_by_status(self, status: str = "DONE", limit: int = 100) -> List[Dict]:
        """
        Read entity linking results filtered by processing status.
        
        Args:
            status: Status to filter by (e.g., "DONE", "TODO", "DOING")
            limit: Maximum number of results to return
            
        Returns:
            List of document results matching the status criteria
        """
        db = self.get_db()
        input_collection = db[self.input_collection]
        
        query = {
            "client_id": self.client_id,
            "dataset_name": self.dataset_name,
            "table_name": self.table_name,
            "status": status
        }
        
        # Only fetch essential fields
        projection = {
            "el_results": 1, 
            "row_id": 1,
            "status": 1,
            "ml_status": 1
        }
        
        cursor = input_collection.find(query, projection).limit(limit).sort("row_id", 1)
        
        results = []
        for doc in cursor:
            result = {
                "row_id": doc["row_id"],
                "status": doc.get("status", "UNKNOWN"),
                "ml_status": doc.get("ml_status", "UNKNOWN"),
                "el_results": doc.get("el_results", {})
            }
            results.append(result)
            
        return results
    
    def get_status_counts(self) -> Dict[str, int]:
        """
        Count documents by their processing status.
        
        Returns:
            Dictionary with counts for each status value
        """
        db = self.get_db()
        input_collection = db[self.input_collection]
        
        base_query = {
            "client_id": self.client_id,
            "dataset_name": self.dataset_name,
            "table_name": self.table_name
        }
        
        # Get counts for each status
        statuses = ["TODO", "DOING", "DONE", "ERROR"]
        counts = {}
        
        for status in statuses:
            query = base_query.copy()
            query["status"] = status
            counts[status] = input_collection.count_documents(query)
            
        # Add total count
        counts["TOTAL"] = input_collection.count_documents(base_query)
        
        return counts
