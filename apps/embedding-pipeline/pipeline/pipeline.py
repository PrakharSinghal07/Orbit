import logging
import time
from typing import Dict, Any, List, Optional

from embedding_pipeline.consumers import KafkaConsumer
from embedding_pipeline.embedders import HuggingFaceEmbedder
from embedding_pipeline.storage import QdrantStorage


class EmbeddingPipeline:
    """
    Main pipeline class that orchestrates the embedding workflow.
    Connects Kafka consumer, embedder, and vector storage.
    """
    
    def __init__(
        self,
        kafka_config: Dict[str, Any],
        kafka_topic: str,
        qdrant_url: str,
        qdrant_collection_name: str,
        embedding_model_name: str,
        vector_size: int = 1024,
        batch_size: int = 10,
        text_field: str = "text",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the pipeline components.
        
        Args:
            kafka_config: Kafka consumer configuration
            kafka_topic: Kafka topic to consume from
            qdrant_url: URL for Qdrant connection
            qdrant_collection_name: Name of the Qdrant collection
            embedding_model_name: Name of the HuggingFace model
            vector_size: Size of embedding vectors
            batch_size: Number of messages to process in a batch
            text_field: Field in the JSON containing text to embed
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initializing Embedding Pipeline")
        
        self.kafka_consumer = KafkaConsumer(kafka_config, kafka_topic, logger=self.logger)
        self.embedder = HuggingFaceEmbedder(embedding_model_name, logger=self.logger)
        self.storage = QdrantStorage(qdrant_url, qdrant_collection_name, vector_size, logger=self.logger)
        
        self.batch_size = batch_size
        self.text_field = text_field
        self.logger.info(f"Pipeline initialized with batch size {batch_size} and text field '{text_field}'")
    
    def process_batch(self, messages: List[Dict[str, Any]]):
        """
        Process a batch of messages.
        
        Args:
            messages: List of messages to process
        """
        start_time = time.time()
        
        texts = []
        filtered_messages = []
        
        for message in messages:
            text = message.get(self.text_field)
            if text:
                texts.append(text)
                filtered_messages.append(message)
            else:
                self.logger.warning(f"No text found in message with field '{self.text_field}'")
        
        if not texts:
            self.logger.warning("No valid texts found in batch")
            return
        
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.embedder.generate_embeddings(texts)
            
            self.logger.info(f"Storing {len(embeddings)} embeddings in Qdrant")
            self.storage.store_embeddings(filtered_messages, embeddings)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Processed batch of {len(texts)} messages in {elapsed:.2f} seconds")
        except Exception as e:
            self.logger.exception(f"Error processing batch: {e}")
    
    def run(self):
        """
        Run the pipeline.
        """
        self.logger.info("Starting embedding pipeline")
        try:
            self.kafka_consumer.consume_batch(self.batch_size, self.process_batch)
        except Exception as e:
            self.logger.exception(f"Error in pipeline execution: {e}")
            raise