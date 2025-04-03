import os
from typing import Dict, Any


class Config:
    """Configuration manager for the embedding pipeline."""
    
    @staticmethod
    def get_kafka_config() -> Dict[str, Any]:
        """Get Kafka configuration from environment variables."""
        return {
            'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
            'group.id': os.getenv('KAFKA_GROUP_ID', 'embedding-pipeline'),
            'auto.offset.reset': os.getenv('KAFKA_AUTO_OFFSET_RESET', 'earliest')
        }
    
    @staticmethod
    def get_kafka_topic() -> str:
        """Get Kafka topic from environment variables."""
        return os.getenv('KAFKA_TOPIC', 'documents-topic')
    
    @staticmethod
    def get_qdrant_config() -> Dict[str, Any]:
        """Get Qdrant configuration from environment variables."""
        host = os.getenv('QDRANT_HOST', 'qdrant')
        port = os.getenv('QDRANT_PORT', '6333')
        
        return {
            'url': f"http://{host}:{port}",
            'collection_name': os.getenv('QDRANT_COLLECTION', 'documents')
        }
    
    @staticmethod
    def get_embedding_config() -> Dict[str, Any]:
        """Get embedding model configuration from environment variables."""
        return {
            'model_name': os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-large-instruct'),
            'text_field': os.getenv('TEXT_FIELD', 'content'),
            'batch_size': int(os.getenv('BATCH_SIZE', '10')),
            'vector_size': int(os.getenv('VECTOR_SIZE', '1024'))
        }

    @staticmethod
    def get_logging_config() -> Dict[str, Any]:
        """Get logging configuration from environment variables."""
        return {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'format': os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        }