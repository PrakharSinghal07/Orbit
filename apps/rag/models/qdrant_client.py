import time
import httpx
from typing import List, Dict, Any, Optional, Callable
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from ..config import settings

class QdrantClientWrapper:
    """Wrapper around the Qdrant client with improved error handling."""
    
    def __init__(
        self, 
        url: str = settings.QDRANT_URL, 
        api_key: str = settings.QDRANT_API_KEY, 
        timeout: int = settings.CONNECTION_TIMEOUT,
        retry_attempts: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the Qdrant client wrapper.
        
        Args:
            url: Qdrant server URL
            api_key: Qdrant API key
            timeout: Connection timeout in seconds
            retry_attempts: Number of retry attempts for operations
            retry_delay: Delay between retries in seconds
        """
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        
    def with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Callable operation to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(self.retry_attempts):
            try:
                return operation(*args, **kwargs)
            except (ResponseHandlingException, httpx.ReadTimeout, ConnectionError) as e:
                if attempt < self.retry_attempts - 1:
                    print(f"Connection error: {e}")
                    print(f"Retrying in {self.retry_delay} seconds... (Attempt {attempt+1}/{self.retry_attempts})")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed after {self.retry_attempts} attempts: {e}")
                    raise
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the Qdrant server.
        
        Returns:
            List of collection names
        """
        collections = self.with_retry(self.client.get_collections).collections
        return [collection.name for collection in collections]
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists, False otherwise
        """
        return collection_name in self.list_collections()
    
    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of vectors to store in the collection
            
        Returns:
            True if successful
        """
        self.with_retry(
            self.client.create_collection,
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        return True
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if successful
        """
        self.with_retry(self.client.delete_collection, collection_name=collection_name)
        return True
    
    def upsert_points(self, collection_name: str, points: List[models.PointStruct], batch_size: int = 100) -> int:
        """
        Insert or update points in a collection.
        
        Args:
            collection_name: Name of the collection
            points: List of points to insert or update
            batch_size: Number of points to insert or update at once
            
        Returns:
            Number of points inserted or updated
        """
        # Upload in batches to avoid payload size limitations
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.with_retry(
                self.client.upsert,
                collection_name=collection_name,
                points=batch
            )
        return len(points)
    
    def search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 10, 
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Search for points in a collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Maximum number of results to return
            filter_conditions: Optional filter conditions for search
            
        Returns:
            List of search results
        """
        search_params = {
            "collection_name": collection_name,
            "query": query_vector,
            "limit": limit,
            "timeout": self.timeout
        }
        
        if filter_conditions:
            search_params["query_filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filter_conditions.items()
                ]
            )
        
        return self.with_retry(self.client.query_points, **search_params).points
