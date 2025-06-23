import json
import time
import threading
from typing import Dict, Any, Optional
from queue import Queue, Empty
from datetime import datetime
import os
from services.utils import log_info, log_error
from services.result_sync import ResultSyncService
from crocodile import Crocodile
import pandas as pd

class TaskQueue:
    """
    Simple in-memory task queue for processing CSV uploads sequentially.
    Prevents resource exhaustion from concurrent uploads.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TaskQueue, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.task_queue = Queue()
            self.processing = False
            self.worker_thread = None
            self.max_concurrent_tasks = 2  # Limit concurrent processing
            self.current_tasks = 0
            self.initialized = True
            self.start_worker()
    
    def add_csv_task(self, task_data: Dict[str, Any]) -> str:
        """Add a CSV processing task to the queue."""
        task_id = f"csv_{int(time.time())}_{task_data['user_id'][:8]}"
        task_data['task_id'] = task_id
        task_data['created_at'] = datetime.now()
        task_data['status'] = 'queued'
        
        self.task_queue.put(task_data)
        log_info(f"Added CSV task {task_id} to queue")
        return task_id
    
    def start_worker(self):
        """Start the background worker thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            log_info("Task queue worker started")
    
    def _worker(self):
        """Background worker that processes tasks from the queue."""
        while True:
            try:
                # Get task with timeout to allow periodic cleanup
                task = self.task_queue.get(timeout=30)
                
                # Check if we can process more tasks
                while self.current_tasks >= self.max_concurrent_tasks:
                    time.sleep(1)
                
                # Process the task in a separate thread to maintain concurrency control
                processing_thread = threading.Thread(
                    target=self._process_csv_task, 
                    args=(task,)
                )
                processing_thread.start()
                
                # Don't wait for completion, just track the count
                self.current_tasks += 1
                
            except Empty:
                # No tasks in queue, continue
                continue
            except Exception as e:
                log_error("Error in task queue worker", e)
                time.sleep(5)
    
    def _process_csv_task(self, task_data: Dict[str, Any]):
        """Process a single CSV task."""
        task_id = task_data.get('task_id')
        try:
            log_info(f"Processing CSV task {task_id}")
            
            # Extract task parameters
            user_id = task_data['user_id']
            dataset_name = task_data['dataset_name']
            table_name = task_data['table_name']
            df = task_data['dataframe']
            classification = task_data['classification']
            
            # Run Crocodile processing
            croco = Crocodile(
                input_csv=df,
                client_id=user_id,
                dataset_name=dataset_name,
                table_name=table_name,
                entity_retrieval_endpoint=os.environ.get("ENTITY_RETRIEVAL_ENDPOINT"),
                entity_retrieval_token=os.environ.get("ENTITY_RETRIEVAL_TOKEN"),
                max_workers=4,  # Reduced from 8 to prevent resource exhaustion
                candidate_retrieval_limit=10,
                model_path="./crocodile/models/default.h5",
                save_output_to_csv=False,
                columns_type=classification,
                entity_bow_endpoint=os.environ.get("ENTITY_BOW_ENDPOINT"),
            )
            croco.run()
            
            # Small delay before starting sync
            time.sleep(2)
            
            # Sync results
            mongo_uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
            sync_service = ResultSyncService(mongo_uri=mongo_uri)
            sync_service.sync_results(
                user_id=user_id, 
                dataset_name=dataset_name, 
                table_name=table_name
            )
            
            log_info(f"CSV task {task_id} completed successfully")
            
        except Exception as e:
            log_error(f"CSV task {task_id} failed", e)
        finally:
            self.current_tasks -= 1

# Global instance
task_queue = TaskQueue()
