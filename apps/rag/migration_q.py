import numpy as np
import time
import re
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import socket
import httpx
import textwrap
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
import google.generativeai as genai
from tqdm import tqdm
from benchmarking import Benchmark, benchmark

# Constants for chunking
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
MAX_CHUNK_SIZE = 2000

def is_port_open(host, port, timeout=2):
    """Check if the given port is open on the host"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

class AgenticChunker:
    """
    Intelligent document chunking that uses LLM to make decisions about
    how to split documents in a context-aware manner.
    """
    
    def __init__(self, gemini_api_key=None, model_name='gemini-1.5-pro'):
        """
        Initialize the agentic chunker with Gemini LLM.
        
        Args:
            gemini_api_key: API key for Gemini
            model_name: Gemini model to use
        """
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        else:
            gemini_api_key = os.environ.get("GEMINI_API_KEY")
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
            else:
                raise ValueError("Gemini API key must be provided or set as GEMINI_API_KEY environment variable")
        
        self.model = genai.GenerativeModel(model_name)
        
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """
        Use Gemini to analyze the document structure and recommend chunking parameters.
        
        Args:
            text: The document text to analyze
            
        Returns:
            Dict with chunking recommendations
        """
        # Take a sample of the document if it's very long
        sample = text[:10000] if len(text) > 10000 else text
        
        prompt = f"""
        Analyze the following document sample and recommend optimal chunking parameters:
        
        Document Sample:
        {sample}
        
        Based on this sample, please determine:
        1. Optimal chunk size (in characters)
        2. Optimal chunk overlap (in characters)
        3. What natural boundaries should be respected (e.g., paragraphs, sections)
        4. If hierarchical chunking would be beneficial
        5. Any other special considerations for this document
        
        Format your response as JSON with the following fields:
        - chunk_size: int
        - chunk_overlap: int
        - respect_boundaries: list of strings (e.g., ["paragraph", "section"])
        - hierarchical: boolean
        - special_considerations: string
        """
        
        response = self.model.generate_content(prompt)
        response_text = response.text
        
        # Extract JSON from response
        try:
            # This is a simple approach - in a robust implementation, you'd want better JSON extraction
            import json
            # Find JSON content (between { and })
            json_match = re.search(r'(\{.*\})', response_text.replace('\n', ' '), re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                recommendations = json.loads(json_str)
            else:
                # Fallback to default values
                recommendations = {
                    "chunk_size": DEFAULT_CHUNK_SIZE,
                    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
                    "respect_boundaries": ["paragraph"],
                    "hierarchical": False,
                    "special_considerations": "None detected"
                }
        except Exception as e:
            print(f"Error parsing LLM recommendation: {e}")
            recommendations = {
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
                "respect_boundaries": ["paragraph"],
                "hierarchical": False,
                "special_considerations": "None detected"
            }
            
        return recommendations
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text in an intelligent way using LLM recommendations.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of dictionaries containing chunks with their metadata
        """
        print("Analyzing document for optimal chunking strategy...")
        recommendations = self.analyze_document(text)
        
        chunk_size = min(recommendations.get("chunk_size", DEFAULT_CHUNK_SIZE), MAX_CHUNK_SIZE)
        chunk_overlap = min(recommendations.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP), chunk_size // 2)
        respect_boundaries = recommendations.get("respect_boundaries", ["paragraph"])
        
        print(f"Chunking with size: {chunk_size}, overlap: {chunk_overlap}")
        print(f"Respecting boundaries: {', '.join(respect_boundaries)}")
        
        chunks = []
        
        # Simple chunking by paragraphs first, then by size
        if "paragraph" in respect_boundaries:
            paragraphs = re.split(r'\n\s*\n', text)
            
            current_chunk = ""
            current_length = 0
            
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if not para:
                    continue
                    
                para_length = len(para)
                
                # If paragraph alone exceeds max chunk size, split it further
                if para_length > chunk_size:
                    if current_chunk:
                        # Add current accumulated chunk
                        chunks.append({
                            "text": current_chunk,
                            "metadata": {
                                **(metadata or {}),
                                "chunk_index": len(chunks),
                                "is_partial": False
                            }
                        })
                        current_chunk = ""
                        current_length = 0
                    
                    # Split long paragraph into smaller chunks
                    words = para.split()
                    sub_chunk = ""
                    for word in words:
                        if len(sub_chunk) + len(word) + 1 > chunk_size:
                            chunks.append({
                                "text": sub_chunk,
                                "metadata": {
                                    **(metadata or {}),
                                    "chunk_index": len(chunks),
                                    "is_partial": True
                                }
                            })
                            sub_chunk = word
                        else:
                            sub_chunk += " " + word if sub_chunk else word
                    
                    if sub_chunk:
                        current_chunk = sub_chunk
                        current_length = len(sub_chunk)
                else:
                    # If adding this paragraph exceeds chunk size, start new chunk
                    if current_length + para_length + 1 > chunk_size:
                        chunks.append({
                            "text": current_chunk,
                            "metadata": {
                                **(metadata or {}),
                                "chunk_index": len(chunks),
                                "is_partial": False
                            }
                        })
                        current_chunk = para
                        current_length = para_length
                    else:
                        # Add to current chunk
                        current_chunk += "\n\n" + para if current_chunk else para
                        current_length += para_length + 2
            
            # Add final chunk
            if current_chunk:
                chunks.append({
                    "text": current_chunk,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": len(chunks),
                        "is_partial": False
                    }
                })
        else:
            # Fallback to simple overlapping chunks
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk_text = text[i:i + chunk_size]
                if len(chunk_text) < 50:  # Skip very small chunks at the end
                    continue
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": len(chunks),
                        "chunk_start": i,
                        "chunk_end": i + len(chunk_text)
                    }
                })
                
        # Use Gemini to generate summaries for larger chunks
        if recommendations.get("hierarchical", False):
            print("Generating summaries for hierarchical representation...")
            for i, chunk in enumerate(chunks):
                if len(chunk["text"]) > 200:
                    summary_prompt = f"Summarize the following text in one sentence:\n\n{chunk['text']}"
                    try:
                        summary = self.model.generate_content(summary_prompt).text
                        chunk["metadata"]["summary"] = summary
                    except Exception as e:
                        print(f"Error generating summary: {e}")
        
        return chunks


@benchmark(runs=3, warmup_runs=1, name="push_data")
def push_data_to_qdrant(
    collection_name="my_collection_new",
    host="localhost",
    port=6333,
    data=None,
    recreate_collection=False,
    model_name='intfloat/multilingual-e5-large-instruct',
    connection_timeout=10,
    retry_attempts=3,
    retry_delay=2,
    gemini_api_key=None,
    use_chunking=True
):
    """
    Push data to a Qdrant collection with improved error handling, connection checking,
    and intelligent chunking.
    
    Args:
        collection_name (str): Name of the collection to push data to
        host (str): Qdrant server host
        port (int): Qdrant server port
        data (list): List of documents/items to push
        recreate_collection (bool): Whether to recreate the collection if it exists
        model_name (str): Name of the SentenceTransformer model to use
        connection_timeout (int): Timeout for server connection in seconds
        retry_attempts (int): Number of connection retry attempts
        retry_delay (int): Delay between retries in seconds
        gemini_api_key (str): API key for Google's Gemini model
        use_chunking (bool): Whether to use agentic chunking
    """
    print(f"Checking connection to Qdrant server at {host}:{port}...")
    
    if not is_port_open(host, port):
        print(f"ERROR: Cannot connect to Qdrant server at {host}:{port}")
        print("Please ensure that:")
        print("1. Qdrant server is running (e.g., via Docker)")
        print("2. The host and port are correct")
        print("3. There are no firewall issues blocking the connection")
        print("\nTo start Qdrant with Docker, you can use:")
        print(f"docker run -p {port}:{port} qdrant/qdrant")
        return None
    
    print(f"Connection to {host}:{port} successful!")
    
    print(f"Loading SentenceTransformer model '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    sample_text = "Sample text for determining vector dimension"
    vector_size = len(model.encode(sample_text))
    print(f"Using model '{model_name}' with vector size: {vector_size}")
    
    client = QdrantClient(host=host, port=port, timeout=connection_timeout)
    
    def with_retry(operation, *args, **kwargs):
        for attempt in range(retry_attempts):
            try:
                return operation(*args, **kwargs)
            except (ResponseHandlingException, httpx.ReadTimeout, ConnectionError) as e:
                if attempt < retry_attempts - 1:
                    print(f"Connection error: {e}")
                    print(f"Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{retry_attempts})")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed after {retry_attempts} attempts: {e}")
                    raise
    
    try:
        collections = with_retry(client.get_collections).collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name in collection_names:
            if recreate_collection:
                print(f"Deleting existing collection '{collection_name}'...")
                with_retry(client.delete_collection, collection_name=collection_name)
                print(f"Collection '{collection_name}' deleted.")
            else:
                print(f"Collection '{collection_name}' already exists.")
        
        if recreate_collection or collection_name not in collection_names:
            print(f"Creating collection '{collection_name}' with vector size {vector_size}...")
            with_retry(
                client.create_collection,
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{collection_name}' created successfully.")
        
        if data is None:
            data = [
                {"id": 1, "text": "Machine learning is fascinating", "metadata": {"category": "tech"}},
                {"id": 2, "text": "Neural networks can solve complex problems", "metadata": {"category": "tech"}},
                {"id": 3, "text": "Vector databases are essential for semantic search", "metadata": {"category": "databases"}}
            ]
        
        # Set up chunker if needed
        chunker = None
        if use_chunking:
            chunker = AgenticChunker(gemini_api_key=gemini_api_key)
        
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
            if chunker and len(text) > DEFAULT_CHUNK_SIZE:
                chunks = chunker.chunk_text(text, item.get("metadata", {}))
                for i, chunk in enumerate(chunks):
                    original_id = item.get('id', item_idx)
                    if isinstance(original_id, str) and original_id.isdigit():
                        original_id = int(original_id)
        
                    # Generate a unique integer ID for the chunk
                    chunk_id = int(f"{original_id}{i:03d}")  # e.g. if original_id=1, chunk_ids will be 1000, 1001, 1002...
        
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
                
                # Create instruction-following version of text for the E5 model
                instruction_text = f"query: {item_text}"
                embedding = model.encode(instruction_text)
                
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
        # Upload in batches to avoid payload size limitations
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            with_retry(
                client.upsert,
                collection_name=collection_name,
                points=batch
            )
        
        print(f"Successfully uploaded {len(points)} points to '{collection_name}'")
        return {"status": "success", "points_uploaded": len(points)}
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def search_with_gemini(
    client, 
    collection_name, 
    query_text, 
    model_name='intfloat/multilingual-e5-large-instruct', 
    limit=5,
    connection_timeout=10,
    gemini_api_key=None,
    rerank=True
):
    """
    Search with the uploaded data and use Gemini to enhance results
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to search
        query_text: Query text
        model_name: Name of the embedding model
        limit: Number of results to return
        connection_timeout: Connection timeout
        gemini_api_key: API key for Gemini
        rerank: Whether to use Gemini to rerank results
    """
    try:
        print(f"\nPerforming search in '{collection_name}' for: '{query_text}'")
        model = SentenceTransformer(model_name)
        
        # Create instruction-following version of query for the E5 model
        instruction_query = f"query: {query_text}"
        query_vector = model.encode(instruction_query).tolist()
        
        # Retrieve more results than requested if reranking
        search_limit = limit * 3 if rerank else limit
        
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=search_limit,
            timeout=connection_timeout
        ).points
        
        if not search_results:
            print("No results found.")
            return []
        
        # Set up Gemini for reranking and enhancement
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        else:
            gemini_api_key = os.environ.get("GEMINI_API_KEY")
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
            else:
                print("Warning: Gemini API key not provided. Skipping reranking.")
                rerank = False
        
        if rerank:
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Prepare context for reranking
            context_list = []
            for idx, result in enumerate(search_results):
                text = result.payload.get('text', '')
                score = result.score
                context_list.append({
                    "id": result.id,
                    "text": text,
                    "original_score": score,
                    "original_rank": idx
                })
            
            # Request reranking from Gemini
            rerank_prompt = f"""
            I need help reranking search results for the query: "{query_text}"
            
            Here are the search results, already sorted by vector similarity:
            
            {json.dumps(context_list, indent=2)}
            
            Please rerank these results based on relevance to the query. Consider:
            1. How well the content addresses the query
            2. The depth and quality of information
            3. The specificity to the query topic
            
            Return a JSON array with the IDs of the top {limit} results in order of relevance.
            Format: [id1, id2, id3, ...]
            """
            
            try:
                rerank_response = model.generate_content(rerank_prompt)
                response_text = rerank_response.text
                
                id_match = re.search(r'(\[.*\])', response_text.replace('\n', ' '), re.DOTALL)
                if id_match:
                    reranked_ids = json.loads(id_match.group(1))
                    
                    # Map back to the original results and get top results
                    id_to_result = {str(result.id): result for result in search_results}
                    reranked_results = []
                    
                    for result_id in reranked_ids:
                        result_id_str = str(result_id)
                        if result_id_str in id_to_result:
                            reranked_results.append(id_to_result[result_id_str])
                            if len(reranked_results) >= limit:
                                break
                    
                    # Fall back to original if reranking returned too few results
                    if reranked_results and len(reranked_results) >= limit / 2:
                        search_results = reranked_results
                    
                    print("Results reranked by Gemini LLM")
                else:
                    # Just use the top from the original results
                    search_results = search_results[:limit]
            except Exception as e:
                print(f"Error during reranking: {e}")
                # Fall back to original top results
                search_results = search_results[:limit]
        else:
            # Use original top results
            search_results = search_results[:limit]
            
        # Print the results
        print(f"Search results:")
        for idx, result in enumerate(search_results[:limit]):
            print(f"Result #{idx+1}")
            print(f"ID: {result.id}, Score: {result.score:.4f}")
            print(f"Text: {textwrap.shorten(result.payload.get('text', ''), width=200, placeholder='...')}")
            metadata = {k: v for k, v in result.payload.items() if k != 'text'}
            if metadata:
                print(f"Metadata: {', '.join([f'{k}: {v}' for k, v in metadata.items()])}")
            print()
            
        return search_results
            
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

def retrieve_and_answer_expanded(
    client,
    collection_name,
    query_text,
    model_name='intfloat/multilingual-e5-large-instruct',
    gemini_api_key=None,
    k=3,
    expand_with_model_knowledge=True
):
    """
    Perform retrieval-augmented generation using Qdrant and Gemini,
    with option to expand answers using Gemini's knowledge.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to search
        query_text: User question
        model_name: Name of the embedding model
        gemini_api_key: API key for Gemini
        k: Number of documents to retrieve
        expand_with_model_knowledge: Whether to allow Gemini to use its own knowledge
    """
    # Configure Gemini
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    else:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        else:
            raise ValueError("Gemini API key must be provided")
    
    # Retrieve relevant documents
    search_results = search_with_gemini(
        client,
        collection_name,
        query_text,
        model_name=model_name,
        limit=k,
        gemini_api_key=gemini_api_key,
        rerank=True
    )
    
    if not search_results:
        if expand_with_model_knowledge:
            # Fall back to Gemini's knowledge if no relevant documents found
            model = genai.GenerativeModel('gemini-1.5-pro')
            prompt = f"Please answer this question using your knowledge: {query_text}"
            response = model.generate_content(prompt)
            return response.text
        else:
            return "I couldn't find any relevant information to answer your question."
    
    # Construct context from search results
    context = "\n\n".join([
        f"Document {i+1}:\n{result.payload.get('text', '')}"
        for i, result in enumerate(search_results)
    ])
    
    # Generate answer with Gemini
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    if expand_with_model_knowledge:
        prompt = f"""
        Answer the following question using both the provided information AND your own knowledge. 
        The provided context contains important information related to the question, but you should 
        expand on it with additional relevant details, examples, or explanations from your knowledge.
        
        Question: {query_text}
        
        Context from database:
        {context}
        
        Please provide a comprehensive answer that integrates the context information with your broader knowledge.
        """
    else:
        # Original RAG approach - restrict to only provided information
        prompt = f"""
        Answer the following question based solely on the provided information. If the information to answer the question is not in the provided context, say "I don't have enough information to answer this question."
        
        Question: {query_text}
        
        Context:
        {context}
        
        Answer:
        """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}"

if __name__ == "__main__":
    # You would need to set this environment variable or pass it as a parameter
    # os.environ["GEMINI_API_KEY"] = "your-gemini-api-key-here"
    
    sample_data = [
        {
            "id": 1, 
            "text": """Machine learning algorithms require large datasets to train effectively. The quality of training data directly impacts model performance. Data preprocessing is a critical step in the ML pipeline that involves cleaning, normalization, and feature engineering.
            
            Common preprocessing tasks include handling missing values, encoding categorical variables, and scaling numerical features. Feature selection and dimensionality reduction can help improve model efficiency and prevent overfitting.
            
            Supervised learning algorithms learn from labeled examples, while unsupervised learning discovers patterns in unlabeled data. Reinforcement learning involves agents learning through interaction with an environment.""", 
            "metadata": {"category": "AI", "difficulty": "intermediate"}
        },
        {
            "id": 2, 
            "text": """Vector databases store embeddings for fast similarity search and retrieval. Unlike traditional databases that excel at exact matches, vector databases are optimized for nearest neighbor search in high-dimensional spaces.
            
            These specialized databases use various indexing strategies such as HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index), or PQ (Product Quantization) to achieve sub-linear search complexity. This makes them ideal for applications like semantic search, recommendation systems, and anomaly detection.
            
            Qdrant, Pinecone, Milvus, and Weaviate are popular vector database options, each with different features and optimization strategies.""", 
            "metadata": {"category": "Databases", "difficulty": "advanced"}
        },
        {
            "id": 3, 
            "text": """Neural networks are inspired by the human brain's structure and function. They consist of interconnected nodes (neurons) organized in layers that process information. The basic building block is the artificial neuron, which receives inputs, applies weights, adds a bias, and passes the result through an activation function.
            
            Deep neural networks contain multiple hidden layers between the input and output layers. This depth allows them to learn increasingly abstract representations of the data. Common types include Convolutional Neural Networks (CNNs) for image processing, Recurrent Neural Networks (RNNs) for sequential data, and Transformers for natural language processing.""", 
            "metadata": {"category": "AI", "difficulty": "beginner"}
        },
    ]
    
    qdrant_host = "localhost"
    qdrant_port = 6333
    collection_name = "documents"
    
    # Set your Gemini API key here or as an environment variable
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    result = push_data_to_qdrant(
        collection_name=collection_name,
        host=qdrant_host,
        port=qdrant_port,
        data=sample_data,
        recreate_collection=True,
        connection_timeout=15,
        model_name='intfloat/multilingual-e5-large-instruct',
        gemini_api_key=gemini_api_key,
        use_chunking=True
    )
    
    if result is not None:
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=15)
        
        # Example of semantic search
        print("\n=== SEMANTIC SEARCH EXAMPLE ===")
        search_with_gemini(
            client, 
            collection_name, 
            "What are neural networks?",
            model_name='intfloat/multilingual-e5-large-instruct',
            gemini_api_key=gemini_api_key,
            rerank=True
        )
        
        # Example of retrieval-augmented generation
        print("\n=== RETRIEVAL-AUGMENTED GENERATION EXAMPLE ===")
        query = "What are neural networks?"
        answer = retrieve_and_answer_expanded(
            client,
            collection_name,
            query,
            model_name='intfloat/multilingual-e5-large-instruct',
            gemini_api_key=gemini_api_key
        )
        print(f"Question: {query}")
        print(f"Answer: {answer}")
    else:
        print("\nData upload failed. Cannot perform search.")