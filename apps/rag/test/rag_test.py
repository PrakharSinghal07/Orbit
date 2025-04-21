import time
import numpy as np
import concurrent.futures
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from benchmarking import Benchmark, benchmark

# Constants
COLLECTION_NAME = "large_rag_benchmark_collection"
MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'
QDRANT_URL = "https://00819855-01e9-4396-a2b5-5a856fe32d73.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.I_YX0wNGh_QrZ9A8gjGs8tCgA1a-AKvQ1vyXVJ_QVrs"
DATA_SIZE = 1000000  # Increased to 1 million vectors
VECTOR_SIZE = 1024  # E5 model vector size
NUM_QUERIES = 30  # Number of test queries
MAX_CONNECTIONS = 10
BATCH_SIZE = 10000  # Increased batch size for faster loading
PARALLEL_QUERIES = 5
RAG_RESULTS_LIMIT = 5


def get_client():
    """Create a client with optimized connection settings"""
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=120,  # Increased timeout for large data operations
        prefer_grpc=True
    )


def generate_sample_data(size, vector_dim):
    """Generate random vectors and text for testing - memory-efficient approach for large datasets"""
    print(f"Generating {size} sample points with {vector_dim} dimensions...")
    
    topics = ["machine learning", "vector databases", "natural language processing", 
            "information retrieval", "knowledge graphs", "semantic search",
            "deep learning", "data science", "artificial intelligence", "neural networks"]
    
    # Generate in smaller chunks to avoid memory issues
    chunk_size = 10000
    for chunk_start in range(0, size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, size)
        chunk_size_actual = chunk_end - chunk_start
        
        # Generate vectors for this chunk
        vectors = np.random.random((chunk_size_actual, vector_dim)).astype(np.float32)
        
        for i in range(chunk_size_actual):
            idx = chunk_start + i
            topic = topics[idx % len(topics)]
            # Creating content that simulates real documents with paragraphs
            text = f"Document {idx}: This is an informational text about {topic}. "
            text += f"It contains multiple paragraphs discussing various aspects of {topic}. "
            text += f"The document provides detailed examples and use cases about {topic}. "
            text += f"Category: {'technical' if idx % 3 == 0 else 'general' if idx % 3 == 1 else 'educational'}"
            
            yield {
                "id": idx,
                "vector": vectors[i].tolist(),
                "payload": {
                    "text": text,
                    "topic": topic,
                    "category": f"category_{idx % 5}",
                    "relevance_score": np.random.uniform(0.5, 1.0)
                }
            }

def setup_collection(client, name, vector_size, with_index=True):
    """Set up a collection with or without HNSW index - simplified version"""
    # Delete collection if exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if name in collection_names:
        print(f"Deleting existing collection '{name}'...")
        client.delete_collection(collection_name=name)
    
    # Create collection with simplified settings
    if with_index:
        print(f"Creating collection '{name}' WITH HNSW index...")
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        # Optionally optimize index settings after creation
        client.update_collection(
            collection_name=name,
            hnsw_config=models.HnswConfigDiff(
                m=32,
                ef_construct=128
            )
        )
    else:
        print(f"Creating collection '{name}' WITHOUT index (brute force)...")
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
    
    return name

def upload_data(client, collection_name, data_generator):
    """Upload data to collection using parallel batches - optimized for large datasets"""
    print(f"Uploading data to '{collection_name}'...")
    
    total_points = 0
    batch = []
    
    # Process the data generator in batches
    for point in tqdm(data_generator, desc="Preparing data", total=DATA_SIZE):
        batch.append(models.PointStruct(
            id=point["id"],
            vector=point["vector"],
            payload=point["payload"]
        ))
        
        # When batch is full, upload it
        if len(batch) >= BATCH_SIZE:
            client.upsert(collection_name=collection_name, points=batch)
            total_points += len(batch)
            print(f"Uploaded {total_points} points so far...")
            batch = []
    
    # Upload any remaining points
    if batch:
        client.upsert(collection_name=collection_name, points=batch)
        total_points += len(batch)
    
    print(f"Successfully uploaded {total_points} points")
    
    # For large datasets, optimization can take time
    print("Collection upload complete")


def generate_rag_queries(model, num_queries):
    """Generate realistic RAG-like query vectors"""
    print(f"Generating {num_queries} RAG-style query vectors...")
    queries = []
    
    # Create different types of queries that mimic real user questions
    query_texts = [
        "What is the difference between semantic search and keyword search?",
        "How do vector embeddings work in information retrieval?",
        "Explain the concept of knowledge graphs in simple terms",
        "What are the best practices for implementing RAG with vector databases?",
        "How can machine learning improve search relevance?",
        "What is the relationship between vector similarity and semantic meaning?",
        "How to evaluate the quality of vector search results?",
        "What are embedding models and how do they convert text to vectors?",
        "Explain the curse of dimensionality in vector search",
        "What is approximate nearest neighbor search?",
        "How does HNSW indexing work?",
        "What are the limitations of vector search?",
        "How to combine keyword search with vector search?",
        "What's the difference between dense and sparse vectors?",
        "How to implement hybrid search with vector databases?",
        "What are the best practices for chunking documents for RAG?",
        "How to measure relevance in vector search results?",
        "What is the role of metadata filtering in vector search?",
        "How to handle out-of-domain queries in RAG systems?",
        "What are embeddings and how are they used in NLP?",
        "How to debug vector search when results are not relevant?",
        "What is the impact of vector dimension on search performance?",
        "How do quantization techniques affect vector search quality?",
        "What are the tradeoffs between index speed and search accuracy?",
        "How to implement re-ranking in vector search pipelines?",
        "What's the difference between euclidean and cosine distance?",
        "How to optimize vector databases for low latency?",
        "What are the best practices for RAG prompt engineering?",
        "How to handle multimodal data in vector databases?",
        "What are the limitations of current embedding models?"
    ]
    
    # Ensure we have enough queries
    while len(query_texts) < num_queries:
        query_texts.append(query_texts[len(query_texts) % len(query_texts)])
    
    # Limit to the requested number
    query_texts = query_texts[:num_queries]
    
    # Batch encode all query texts at once for better performance
    vectors = model.encode(query_texts)
    
    for i, vector in enumerate(vectors):
        queries.append({"text": query_texts[i], "vector": vector.tolist()})
    
    return queries


def parallel_search(client, collection_name, query_vectors, limit=RAG_RESULTS_LIMIT, with_index=True):
    """Perform searches in parallel batches"""
    def search_single(query):
        # Set different search parameters based on indexed vs non-indexed search
        if with_index:
            search_params = models.SearchParams(
                hnsw_ef=128,
                exact=False
            )
        else:
            search_params = models.SearchParams(
                exact=True
            )
        
        try:
            start_time = time.time()
            search_results = client.query_points(
                collection_name=collection_name,
                query=query["vector"],
                limit=limit,
                search_params=search_params,
                score_threshold=0.7
            )
            end_time = time.time()
            elapsed = end_time - start_time
            
            return {
                "query": query["text"],
                "time": elapsed,
                "num_results": len(search_results.points)
            }
        except Exception as e:
            print(f"Error during search: {e}")
            return {
                "query": query["text"],
                "time": 0,
                "num_results": 0,
                "error": str(e)
            }
    
    results = []
    total_time = 0
    successful_queries = 0
    
    # Split queries into batches for parallel execution
    desc = f"Searching with {'index' if with_index else 'brute force'}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_QUERIES) as executor:
        futures = {}
        for query in query_vectors:
            futures[executor.submit(search_single, query)] = query["text"]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            results.append(result)
            
            if "error" not in result:
                total_time += result["time"]
                successful_queries += 1
    
    if successful_queries > 0:
        avg_time = total_time / successful_queries
        print(f"Average search time with {'index' if with_index else 'brute force'}: {avg_time:.6f} seconds")
    else:
        print("No successful queries to calculate average time.")
    
    return {
        "collection": collection_name,
        "avg_time": total_time / successful_queries if successful_queries > 0 else 0,
        "results": results,
        "successful_queries": successful_queries
    }


def search_with_index(client, collection_name, query_vectors, limit=RAG_RESULTS_LIMIT):
    """Perform search with indexed collection"""
    return parallel_search(client, collection_name, query_vectors, limit, with_index=True)


def search_without_index(client, collection_name, query_vectors, limit=RAG_RESULTS_LIMIT):
    """Perform search with non-indexed collection"""
    return parallel_search(client, collection_name, query_vectors, limit, with_index=False)


def run_benchmark():
    # Initialize model and client
    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = get_client()
    
    # Set up collections - one with index, one without
    indexed_collection = setup_collection(client, f"{COLLECTION_NAME}_indexed", VECTOR_SIZE, with_index=True)
    non_indexed_collection = setup_collection(client, f"{COLLECTION_NAME}_non_indexed", VECTOR_SIZE, with_index=False)
    
    # Upload data to both collections
    print("Uploading data to indexed collection...")
    # Generate a new generator for the first collection
    data_gen1 = generate_sample_data(DATA_SIZE, VECTOR_SIZE)
    upload_data(client, indexed_collection, data_gen1)
    
    print("Uploading data to non-indexed collection...")
    # Generate a new generator for the second collection
    data_gen2 = generate_sample_data(DATA_SIZE, VECTOR_SIZE)
    upload_data(client, non_indexed_collection, data_gen2)
    
    # Generate query vectors - RAG-style
    query_vectors = generate_rag_queries(model, NUM_QUERIES)
    
    # Pre-warm collections to ensure fair comparison
    print("Pre-warming collections...")
    warm_query = query_vectors[0]
    for _ in range(3):
        try:
            client.query_points(
                collection_name=indexed_collection,
                query=warm_query["vector"],
                limit=RAG_RESULTS_LIMIT,
                search_params=models.SearchParams(hnsw_ef=128, exact=False)
            )
            
            client.query_points(
                collection_name=non_indexed_collection,
                query=warm_query["vector"],
                limit=RAG_RESULTS_LIMIT,
                search_params=models.SearchParams(exact=True)
            )
        except Exception as e:
            print(f"Error during warm-up: {e}")
            time.sleep(1)
    
    # Prepare parameters for benchmarking functions
    indexed_params = {
        "client": client,
        "collection_name": indexed_collection,
        "query_vectors": query_vectors
    }
    
    non_indexed_params = {
        "client": client,
        "collection_name": non_indexed_collection,
        "query_vectors": query_vectors
    }
    
    # Create benchmark instance and run comparison
    print("\nRunning RAG benchmark with large dataset (1M vectors)...")
    bench = Benchmark(name="Large-Scale Qdrant RAG Search Comparison", runs=3, warmup_runs=1)
    
    try:
        comparison = bench.compare(
            funcs=[
                (search_with_index, indexed_params),
                (search_without_index, non_indexed_params)
            ],
            labels=["HNSW Indexed Search (Top 5)", "Brute Force Search (Top 5)"],
            plot=True
        )
        
        # Calculate latency percentiles
        if comparison and len(comparison) >= 2:
            if len(comparison[0]["results"]) > 0 and len(comparison[1]["results"]) > 0:
                indexed_times = [r["time"] for r in comparison[0]["results"][0]["results"] if "error" not in r]
                non_indexed_times = [r["time"] for r in comparison[1]["results"][0]["results"] if "error" not in r]
                
                if indexed_times and non_indexed_times:
                    indexed_p95 = np.percentile(indexed_times, 95)
                    non_indexed_p95 = np.percentile(non_indexed_times, 95)
                    
                    print("\nLatency Analysis:")
                    print(f"Indexed P95 latency: {indexed_p95:.6f} seconds")
                    print(f"Non-indexed P95 latency: {non_indexed_p95:.6f} seconds")
                    
                    print(f"\nSpeed improvement: {(np.mean(non_indexed_times) / np.mean(indexed_times)):.2f}x faster with indexing")
        
        return comparison
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return None


if __name__ == "__main__":
    results = run_benchmark()
    print("\nLarge-scale RAG benchmark comparison complete!")
    if results:
        print("Check the saved plot: 'Large-Scale_Qdrant_RAG_Search_Comparison_comparison.png'")
    else:
        print("Benchmark did not complete successfully.")