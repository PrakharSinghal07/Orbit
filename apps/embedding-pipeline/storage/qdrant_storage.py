import logging
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantStorage:
    """Storage module for managing embeddings in Qdrant vector database."""
    
    def __init__(
        self, 
        url: str, 
        collection_name: str, 
        vector_size: int = 1024,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize connection to Qdrant and ensure collection exists.
        
        Args:
            url: URL of the Qdrant server
            collection_name: Name of the collection to use
            vector_size: Size of embedding vectors
            logger: Logger instance
        """
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"Connecting to Qdrant at {url}")
        self.client = QdrantClient(url=url)
        
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, create it if it doesn't."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.logger.info(f"Creating new collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                self.logger.info(f"Collection {self.collection_name} created successfully")
            else:
                self.logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            self.logger.error(f"Error checking/creating collection: {e}")
            raise
    
    def store_embeddings(self, data: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Store documents with their embeddings in Qdrant.
        
        Args:
            data: List of documents to store
            embeddings: List of embedding vectors corresponding to documents
        """
        if not data or not embeddings:
            self.logger.warning("No data or embeddings provided")
            return
            
        if len(data) != len(embeddings):
            self.logger.error(f"Mismatch between data length ({len(data)}) and embeddings length ({len(embeddings)})")
            return
        
        points = []
        
        for i, (document, embedding) in enumerate(zip(data, embeddings)):
            points.append(models.PointStruct(
                id=document.get("id", i),
                vector=embedding,
                payload=document  
            ))
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            self.logger.info(f"Successfully uploaded {len(points)} embeddings to Qdrant")
        except Exception as e:
            self.logger.error(f"Error uploading embeddings to Qdrant: {e}")