import os
import sys
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from embedding_pipeline.configs.config import Config
from embedding_pipeline.pipeline import EmbeddingPipeline
from embedding_pipeline.utils.logging import setup_logger


def main():
    """Main entry point for the embedding pipeline."""
    # Set up logging
    logger = setup_logger("embedding_pipeline", Config.get_logging_config())
    logger.info("Starting embedding pipeline application")
    
    try:
        kafka_config = Config.get_kafka_config()
        kafka_topic = Config.get_kafka_topic()
        qdrant_config = Config.get_qdrant_config()
        embedding_config = Config.get_embedding_config()
        
        logger.info(f"Kafka topic: {kafka_topic}")
        logger.info(f"Qdrant collection: {qdrant_config['collection_name']}")
        logger.info(f"Embedding model: {embedding_config['model_name']}")
        
        pipeline = EmbeddingPipeline(
            kafka_config=kafka_config,
            kafka_topic=kafka_topic,
            qdrant_url=qdrant_config['url'],
            qdrant_collection_name=qdrant_config['collection_name'],
            embedding_model_name=embedding_config['model_name'],
            vector_size=embedding_config['vector_size'],
            batch_size=embedding_config['batch_size'],
            text_field=embedding_config['text_field'],
            logger=logger
        )
        
        # Run pipeline
        logger.info("Starting pipeline execution")
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    except Exception as e:
        logger.exception(f"Error in pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()