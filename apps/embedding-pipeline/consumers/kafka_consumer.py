import json
import logging
from typing import Dict, Any, List, Callable, Optional
from confluent_kafka import Consumer


class KafkaConsumer:
    """Consumer module for reading messages from Kafka."""
    
    def __init__(self, config: Dict[str, Any], topic: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the Kafka consumer.
        
        Args:
            config: Kafka consumer configuration
            topic: Kafka topic to consume from
            logger: Logger instance
        """
        self.consumer = Consumer(config)
        self.consumer.subscribe([topic])
        self.logger = logger or logging.getLogger(__name__)
        self.topic = topic
        self.logger.info(f"Initialized Kafka consumer for topic: {topic}")
    
    def consume_batch(self, batch_size: int, process_batch: Callable[[List[Dict[str, Any]]], None]):
        """
        Consume messages in batches and process them.
        
        Args:
            batch_size: Size of the batch to collect before processing
            process_batch: Callback function to process the batch
        """
        batch = []
        
        try:
            self.logger.info(f"Starting to consume messages from topic: {self.topic}")
            while True:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    if batch:
                        self.logger.debug(f"Processing batch of {len(batch)} messages")
                        process_batch(batch)
                        batch = []
                    continue
                
                if msg.error():
                    self.logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                try:
                    value = json.loads(msg.value().decode('utf-8'))
                    batch.append(value)
                    
                    if len(batch) >= batch_size:
                        self.logger.debug(f"Processing batch of {len(batch)} messages")
                        process_batch(batch)
                        batch = []
                        
                except json.JSONDecodeError:
                    self.logger.error(f"Error decoding JSON from message: {msg.value()}")
                except Exception as e:
                    self.logger.exception(f"Error processing message: {e}")
        
        except KeyboardInterrupt:
            if batch:
                self.logger.info(f"Processing remaining {len(batch)} messages before shutdown")
                process_batch(batch)
            self.logger.info("Consumer interrupted by user")
        
        finally:
            self.logger.info("Closing Kafka consumer")
            self.consumer.close()