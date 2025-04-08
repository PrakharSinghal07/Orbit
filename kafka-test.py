from confluent_kafka import Consumer
import json

KAFKA_CONFIG = {
    'bootstrap.servers': 'localhost:9092',         # Matches PLAINTEXT_HOST in docker-compose
    'group.id': 'filepulse-consumer-group',
    'auto.offset.reset': 'earliest'
}

TOPIC = 'json-data'

def main():
    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe([TOPIC])

    print(f"🚀 Consuming messages from topic: {TOPIC}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                print(f"❌ Kafka error: {msg.error()}")
                continue

            try:
                raw_value = msg.value().decode("utf-8")
                data = json.loads(raw_value)
                print("📦 New message:\n", json.dumps(data, indent=2))

            except json.JSONDecodeError:
                print("⚠️ JSON decode error:", raw_value)
            except Exception as e:
                print("⚠️ Unexpected error:", e)

    except KeyboardInterrupt:
        print("\n🛑 Stopping consumer...")

    finally:
        consumer.close()
        print("✅ Consumer closed.")

if __name__ == "__main__":
    main()
