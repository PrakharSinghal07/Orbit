import logging
from typing import List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModel


class HuggingFaceEmbedder:
    """Module for generating embeddings using HuggingFace models."""
    
    def __init__(self, model_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the embedder with a HuggingFace model.
        
        Args:
            model_name: Name of the HuggingFace model to use
            logger: Logger instance
        """
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger.info(f"Model loaded and using device: {self.device}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for a given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        self.logger.debug(f"Generating embedding for text of length: {len(text)}")
        
        instruction_text = f"query: {text}"
        
        inputs = self.tokenizer(
            instruction_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings[0].cpu().tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        self.logger.debug(f"Generating embeddings for {len(texts)} texts")
        
        if not texts:
            self.logger.warning("Empty text list provided for embedding generation")
            return []
        
        instruction_texts = [f"query: {text}" for text in texts]
        
        inputs = self.tokenizer(
            instruction_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings.cpu().tolist()
    
    def _mean_pooling(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)