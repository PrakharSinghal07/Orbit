from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
import sys
import os


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import settings

class EmbeddingModel:
    """Class for handling text embeddings."""
    
    def __init__(self, model_name=settings.DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.vector_size = None
        self._determine_vector_size()
        
    def _determine_vector_size(self):
        """Determine the vector size of the embedding model."""
        sample_text = "Sample text for determining vector dimension"
        vector = self.encode(sample_text)
        self.vector_size = len(vector)
        
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into embedding vectors.
        
        Args:
            text: Single text or list of texts to encode
            
        Returns:
            numpy.ndarray: Vector representation(s) of the text(s)
        """
        if isinstance(text, str):
            # Create instruction-following version of text for E5 models
            if "e5" in self.model_name.lower():
                instruction_text = f"query: {text}"
            else:
                instruction_text = text
            return self.model.encode(instruction_text)
        elif isinstance(text, list):
            # Create instruction-following version for each text in the list
            if "e5" in self.model_name.lower():
                instruction_texts = [f"query: {t}" for t in text]
            else:
                instruction_texts = text
            return self.model.encode(instruction_texts)
        else:
            raise ValueError("Text must be a string or list of strings")
            
    def get_vector_size(self) -> int:
        """Get the vector size of the embedding model."""
        return self.vector_size