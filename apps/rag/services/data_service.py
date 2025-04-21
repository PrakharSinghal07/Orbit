import os
import sys
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Replace relative imports with absolute imports
from models.chunker import AgenticChunker
from models.embeddings import EmbeddingModel
from models.qdrant_client import QdrantClientWrapper
from qdrant_client.http import models
from config import settings

class DataService:
    """Service for managing data in Qdrant."""
    
    def __init__(
        self,
        qdrant_client: QdrantClientWrapper = None,
        embedding_model: EmbeddingModel = None,
        chunker: AgenticChunker = None
    ):
        """
        Initialize the data service.
        
        Args:
            qdrant_client: Qdrant client wrapper
            embedding_model: Embedding model
            chunker: Document chunker
        """
        self.qdrant_client = qdrant_client or QdrantClientWrapper()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.chunker = chunker
        
    def initialize_chunker(self, gemini_api_key: Optional[str] = None) -> AgenticChunker:
        """
        Initialize the document chunker if not already initialized.
        
        Args:
            gemini_api_key: API key for Gemini
            
        Returns:
            Initialized chunker
        """
        if self.chunker is None:
            self.chunker = AgenticChunker(gemini_api_key=gemini_api_key)
        return self.chunker
    
    def push_data(
        self,
        data: List[Dict[str, Any]],
        collection_name: str = settings.DEFAULT_COLLECTION_NAME,
        recreate_collection: bool = False,
        use_chunking: bool = True,
        gemini_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Push data to Qdrant.
        
        Args:
            data: List of data items to push
            collection_name: Name of the collection to push data to
            recreate_collection: Whether to recreate the collection if it exists
            use_chunking: Whether to use intelligent chunking
            gemini_api_key: API key for Gemini
            
        Returns:
            Dictionary with push results
        """
        # Initialize chunker if chunking is enabled
        if use_chunking:
            self.initialize_chunker(gemini_api_key)
        
        vector_size = self.embedding_model.get_vector_size()
        print(f"Using model '{self.embedding_model.model_name}' with vector size: {vector_size}")
        
        # Check if collection exists and handle recreation if needed
        if self.qdrant_client.collection_exists(collection_name):
            if recreate_collection:
                print(f"Deleting existing collection '{collection_name}'...")
                self.qdrant_client.delete_collection(collection_name)
                print(f"Collection '{collection_name}' deleted.")
                print(f"Creating collection '{collection_name}' with vector size {vector_size}...")
                self.qdrant_client.create_collection(collection_name, vector_size)
                print(f"Collection '{collection_name}' created successfully.")
            else:
                print(f"Collection '{collection_name}' already exists.")
        else:
            print(f"Creating collection '{collection_name}' with vector size {vector_size}...")
            self.qdrant_client.create_collection(collection_name, vector_size)
            print(f"Collection '{collection_name}' created successfully.")
        
        print(f"Preparing embeddings for {len(data)} items...")
        points = []
        
        for item_idx, item in enumerate(tqdm(data)):
            text = item.get("text", "")
            
            # Skip empty texts
            if not text.strip():
                print(f"Skipping item {item_idx} - empty text")
                continue
            
            # Apply chunking if enabled and text is long enough
            processed_items = []
            if use_chunking and self.chunker and len(text) > settings.DEFAULT_CHUNK_SIZE:
                chunks = self.chunker.chunk_text(text, item.get("metadata", {}))
                for i, chunk in enumerate(chunks):
                    original_id = item.get('id', item_idx)
                    if isinstance(original_id, str) and original_id.isdigit():
                        original_id = int(original_id)
        
                    # Generate a unique integer ID for the chunk
                    # e.g. if original_id=1, chunk_ids will be 1000, 1001, 1002...
                    chunk_id = int(f"{original_id}{i:03d}")
        
                    # Create a new item for each chunk
                    chunk_item = {
                        "id": chunk_id,
                        "text": chunk["text"],
                        "metadata": {
                            **chunk["metadata"],
                            "parent_id": original_id,
                            "chunk_index": i,
                            "original_text_length": len(text)
                        }
                    }
                    processed_items.append(chunk_item)
            else:
                # If chunking is disabled or text is short enough, use the original item
                item_id = item.get("id", item_idx)
                if isinstance(item_id, str) and item_id.isdigit():
                    item_id = int(item_id)  # Convert string digits to int
    
                processed_items.append({
                    "id": item_id,
                    "text": text,
                    "metadata": item.get("metadata", {})
                })
            
            # Embed and create points for each processed item
            for proc_item in processed_items:
                item_text = proc_item["text"]
                embedding = self.embedding_model.encode(item_text)
                
                # Create the point with proper ID handling
                point_id = proc_item["id"]
                if isinstance(point_id, str) and point_id.isdigit():
                    point_id = int(point_id)
                
                point = models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": item_text,
                        **proc_item.get("metadata", {})
                    }
                )
                points.append(point)
        
        print(f"Uploading {len(points)} points to collection '{collection_name}'...")
        
        points_uploaded = self.qdrant_client.upsert_points(collection_name, points)
        
        print(f"Successfully uploaded {points_uploaded} points to '{collection_name}'")
        return {"status": "success", "points_uploaded": points_uploaded}