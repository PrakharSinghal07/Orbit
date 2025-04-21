import time
import numpy as np
import concurrent.futures
from qdrant_client import QdrantClient, grpc
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from benchmarking import Benchmark, benchmark

# Constants
COLLECTION_NAME = "benchmark_collection"
MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'
QDRANT_URL = "https://00819855-01e9-4396-a2b5-5a856fe32d73.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.I_YX0wNGh_QrZ9A8gjGs8tCgA1a-AKvQ1vyXVJ_QVrs"
DATA_SIZE = 1000000  # Number of vectors to generate
VECTOR_SIZE = 1024  # E5 model vector size
NUM_QUERIES = 20  # Number of queries to run for benchmarking
MAX_CONNECTIONS = 10  # Maximum number of concurrent connections
BATCH_SIZE = 1000  # Increased batch size for uploads
PARALLEL_QUERIES = 5  # Number of parallel queries to run


def get_client():
    """Create a client with optimized connection settings"""
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,  # Increased timeout
        prefer_grpc=True,  # Use gRPC for better performance
        # Default connection pooling is provided by the client itself
    )

def generate_sample_data(size, vector_dim):
    """Generate random vectors and text for testing"""
    print(f"Generating {size} sample points with {vector_dim} dimensions...")
    data = []
    # Use more efficient numpy operations for generating data
    all_vectors = np.random.random((size, vector_dim)).astype(np.float32)
    
    for i in range(size):
        data.append({
            "id": i,
            "vector": all_vectors[i].tolist(),
            "payload": {"text": f"Sample document {i} with random content for testing search performance", 
                       "category": f"category_{i % 5}"}
        })
    return data


def setup_collection(client, name, vector_size, with_index=True):
    """Set up a collection with or without HNSW index"""
    # Delete collection if exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if name in collection_names:
        print(f"Deleting existing collection '{name}'...")
        client.delete_collection(collection_name=name)
    
    # Create collection with specified index settings
    if with_index:
        print(f"Creating collection '{name}' WITH HNSW index...")
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=False,  # In-memory for best performance
                hnsw_config=models.HnswConfigDiff(
                    m=32,  # Increased from 16 for better graph connectivity
                    ef_construct=200,  # Increased from 100 for better index quality
                )
            ),
            optimizers_config=models.OptimizersConfigDiff(
                memmap_threshold=100000,  # Optimize for large collections
                indexing_threshold=20000,  # Start indexing earlier
                flush_interval_sec=5,  # Flush to disk more frequently
            ),
        )
    else:
        print(f"Creating collection '{name}' WITHOUT index (brute force)...")
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=False,  # In-memory for best performance
                # No HNSW config means brute force search
            )
        )
    
    return name


def upload_data(client, collection_name, data):
    """Upload data to collection using parallel batches"""
    print(f"Uploading {len(data)} points to '{collection_name}'...")
    
    # Use larger batch size for more efficient uploads
    batch_size = BATCH_SIZE
    batches = []
    
    for i in range(0, len(data), batch_size):
        batch = [
            models.PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point["payload"]
            ) for point in data[i:i + batch_size]
        ]
        batches.append(batch)
    
    # Upload batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONNECTIONS) as executor:
        futures = []
        for batch in batches:
            futures.append(executor.submit(client.upsert, collection_name=collection_name, points=batch))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Uploading batches"):
            future.result()  # This will raise any exceptions that occurred

    # Wait for optimization
    print("Optimizing collection...")
    client.update_collection(
        collection_name=collection_name,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0  # Force immediate indexing
        )
    )
    
    # Wait for the collection to be fully optimized
    while True:
        collection_info = client.get_collection(collection_name=collection_name)
        if collection_info.status == "green":
            break
        print("Waiting for collection optimization... Current status:", collection_info.status)
        time.sleep(2)
    
    print(f"Successfully uploaded {len(data)} points")


def generate_query_vectors(model, num_queries):
    """Generate query vectors using the model"""
    print(f"Generating {num_queries} query vectors...")
    queries = []
    
    # Create different types of queries for more realistic benchmarking
    query_texts = []
    for i in range(num_queries):
        if i % 3 == 0:
            query_texts.append(f"query: What is the best example of category {i % 5}?")
        elif i % 3 == 1:
            query_texts.append(f"query: Find documents related to sample {i * 10}")
        else:
            query_texts.append(f"query: Show me items with random content for testing")
    
    # Batch encode all query texts at once for better performance
    vectors = model.encode(query_texts)
    
    for i, vector in enumerate(vectors):
        queries.append({"text": query_texts[i], "vector": vector.tolist()})
    
    return queries


def parallel_search(client, collection_name, query_vectors, limit=10, with_index=True):
    """Perform searches in parallel batches"""
    def search_single(query):
        # Set different search parameters based on indexed vs non-indexed search
        if with_index:
            search_params = models.SearchParams(
                hnsw_ef=200,  # Higher ef value for better search quality
                exact=False
            )
        else:
            search_params = models.SearchParams(
                exact=True  # Ensure brute force search for non-indexed collection
            )
        
        start_time = time.time()
        search_results = client.query_points(
            collection_name=collection_name,
            query=query["vector"],
            limit=limit,
            search_params=search_params
        )
        end_time = time.time()
        elapsed = end_time - start_time
        
        return {
            "query": query["text"],
            "time": elapsed,
            "num_results": len(search_results.points)
        }
    
    results = []
    total_time = 0
    
    # Split queries into batches for parallel execution
    desc = f"Searching with {'index' if with_index else 'brute force'}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_QUERIES) as executor:
        futures = []
        for query in query_vectors:
            futures.append(executor.submit(search_single, query))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            results.append(result)
            total_time += result["time"]
    
    avg_time = total_time / len(query_vectors)
    print(f"Average search time with {'index' if with_index else 'brute force'}: {avg_time:.6f} seconds")
    
    return {
        "collection": collection_name,
        "avg_time": avg_time,
        "results": results
    }


def search_with_index(client, collection_name, query_vectors, limit=10):
    """Perform search with indexed collection"""
    return parallel_search(client, collection_name, query_vectors, limit, with_index=True)


def search_without_index(client, collection_name, query_vectors, limit=10):
    """Perform search with non-indexed collection"""
    return parallel_search(client, collection_name, query_vectors, limit, with_index=False)


def run_benchmark():
    # Initialize model and client
    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = get_client()
    
    # Generate sample data
    data = generate_sample_data(DATA_SIZE, VECTOR_SIZE)
    
    # Set up collections - one with index, one without
    indexed_collection = setup_collection(client, f"{COLLECTION_NAME}_indexed", VECTOR_SIZE, with_index=True)
    non_indexed_collection = setup_collection(client, f"{COLLECTION_NAME}_non_indexed", VECTOR_SIZE, with_index=False)
    
    # Upload data to both collections
    upload_data(client, indexed_collection, data)
    upload_data(client, non_indexed_collection, data)
    
    # Generate query vectors
    query_vectors = generate_query_vectors(model, NUM_QUERIES)
    
    # Pre-warm collections to ensure fair comparison
    print("Pre-warming collections...")
    warm_query = query_vectors[0]
    for _ in range(3):
        client.query_points(
            collection_name=indexed_collection,
            query=warm_query["vector"],
            limit=10,
            search_params=models.SearchParams(hnsw_ef=200, exact=False)
        )
        client.query_points(
            collection_name=non_indexed_collection,
            query=warm_query["vector"],
            limit=10,
            search_params=models.SearchParams(exact=True)
        )
    
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
    bench = Benchmark(name="Qdrant Search Comparison", runs=3, warmup_runs=2)
    comparison = bench.compare(
        funcs=[
            (search_with_index, indexed_params),
            (search_without_index, non_indexed_params)
        ],
        labels=["HNSW Indexed Search", "Brute Force Search"],
        plot=True
    )
    
    # Run additional benchmark with larger limit to see how results scale
    print("\nRunning additional benchmark with larger result set (limit=100)...")
    indexed_params["limit"] = 100
    non_indexed_params["limit"] = 100
    
    bench_large = Benchmark(name="Qdrant Search Comparison (Large Result Set)", runs=2, warmup_runs=1)
    comparison_large = bench_large.compare(
        funcs=[
            (search_with_index, indexed_params),
            (search_without_index, non_indexed_params)
        ],
        labels=["HNSW Indexed Search (100 results)", "Brute Force Search (100 results)"],
        plot=True
    )
    
    return comparison, comparison_large


if __name__ == "__main__":
    results, results_large = run_benchmark()
    print("\nBenchmark comparison complete!")
    print("Check the saved plots: 'Qdrant_Search_Comparison_comparison.png' and 'Qdrant_Search_Comparison_(Large_Result_Set)_comparison.png'")