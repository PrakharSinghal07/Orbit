from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .routers import data_router, search_router, rag_router
import os
import uvicorn
from ..config import settings

app = FastAPI(
    title="RAG API",
    description="API for Retrieval-Augmented Generation with intelligent chunking",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router.router)
app.include_router(search_router.router)
app.include_router(rag_router.router)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG API",
        "version": "0.1.0",
        "description": "API for Retrieval-Augmented Generation with intelligent chunking",
        "endpoints": {
            "data": "/data/push",
            "search": "/search",
            "rag": "/rag/answer"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("rag_api.api.main:app", host="0.0.0.0", port=port, reload=True)

import os
import json
import requests

BASE_URL = "http://localhost:8000"

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

def push_data():
    """Push data to the vector database."""
    url = f"{BASE_URL}/data/push"
    
    payload = {
        "collection_name": "documents",
        "data": sample_data,
        "recreate_collection": True,
        "use_chunking": True,
        "gemini_api_key": os.environ.get("GEMINI_API_KEY", "")
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Data pushed successfully!")
        print(json.dumps(response.json(), indent=2))
        return True
    else:
        print(f"Failed to push data: {response.status_code}")
        print(response.text)
        return False

def search_documents():
    """Search for documents in the vector database."""
    url = f"{BASE_URL}/search"
    
    payload = {
        "query": "What are neural networks?",
        "collection_name": "documents",
        "limit": 3,
        "rerank": True,
        "gemini_api_key": os.environ.get("GEMINI_API_KEY", "")
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Search results:")
        print(json.dumps(response.json(), indent=2))
        return True
    else:
        print(f"Failed to search: {response.status_code}")
        print(response.text)
        return False

def retrieve_and_answer():
    """Get an answer to a question using RAG."""
    url = f"{BASE_URL}/rag/answer"
    
    payload = {
        "query": "How do neural networks work and what are their applications?",
        "collection_name": "documents",
        "k": 3,
        "expand_with_model_knowledge": True,
        "gemini_api_key": os.environ.get("GEMINI_API_KEY", "")
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Retrieved {len(result['retrieved_documents'])} documents")
        return True
    else:
        print(f"Failed to get answer: {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("Please set the GEMINI_API_KEY environment variable")
        exit(1)
    
    if push_data():
        print("\n" + "-" * 50 + "\n")
        search_documents()
        print("\n" + "-" * 50 + "\n")
        retrieve_and_answer()