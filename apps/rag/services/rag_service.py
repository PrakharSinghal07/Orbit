import os
import sys
import google.generativeai as genai
from typing import Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.qdrant_client import QdrantClientWrapper
from models.embeddings import EmbeddingModel
from .search_service import SearchService

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from config import settings

class RAGService:
    """Service for Retrieval-Augmented Generation."""
    
    def __init__(
        self,
        search_service: SearchService = None,
        qdrant_client: QdrantClientWrapper = None,
        embedding_model: EmbeddingModel = None
    ):
        """
        Initialize the RAG service.
        
        Args:
            search_service: Search service
            qdrant_client: Qdrant client wrapper
            embedding_model: Embedding model
        """
        if search_service:
            self.search_service = search_service
        else:
            qdrant_client = qdrant_client or QdrantClientWrapper()
            embedding_model = embedding_model or EmbeddingModel()
            self.search_service = SearchService(qdrant_client, embedding_model)
    
    def retrieve_and_answer(
        self,
        query_text: str,
        collection_name: str = settings.DEFAULT_COLLECTION_NAME,
        gemini_api_key: Optional[str] = None,
        k: int = 3,
        expand_with_model_knowledge: bool = True,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform retrieval-augmented generation.
        
        Args:
            query_text: User question
            collection_name: Name of the collection to search
            gemini_api_key: API key for Gemini
            k: Number of documents to retrieve
            expand_with_model_knowledge: Whether to allow the model to use its knowledge
            filter_conditions: Optional filter conditions for search
            
        Returns:
            Dictionary with answer and context
        """
        # Configure Gemini
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        else:
            gemini_api_key = os.environ.get("GEMINI_API_KEY", settings.GEMINI_API_KEY)
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
            else:
                raise ValueError("Gemini API key must be provided")
        
        # Retrieve relevant documents
        search_results = self.search_service.search(
            query_text,
            collection_name=collection_name,
            limit=k,
            gemini_api_key=gemini_api_key,
            rerank=True,
            filter_conditions=filter_conditions
        )
        
        if not search_results:
            if expand_with_model_knowledge:
                # Fall back to Gemini's knowledge if no relevant documents found
                model = genai.GenerativeModel('gemini-1.5-pro')
                prompt = f"Please answer this question using your knowledge: {query_text}"
                response = model.generate_content(prompt)
                return {
                    "question": query_text,
                    "answer": response.text,
                    "retrieved_documents": [],
                    "used_model_knowledge": True
                }
            else:
                return {
                    "question": query_text,
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "retrieved_documents": [],
                    "used_model_knowledge": False
                }
        
        # Construct context from search results
        documents = []
        context = ""
        for i, result in enumerate(search_results):
            doc_text = result.payload.get('text', '')
            context += f"Document {i+1}:\n{doc_text}\n\n"
            
            # Add document to return object
            documents.append({
                "id": result.id,
                "text": doc_text,
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k != 'text'}
            })
        
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
            Answer the following question based solely on the provided information. If the information 
            to answer the question is not in the provided context, say "I don't have enough information to answer this question."
            
            Question: {query_text}
            
            Context:
            {context}
            
            Answer:
            """
        
        try:
            response = model.generate_content(prompt)
            return {
                "question": query_text,
                "answer": response.text,
                "retrieved_documents": documents,
                "used_model_knowledge": expand_with_model_knowledge
            }
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "question": query_text,
                "answer": f"Error generating answer: {str(e)}",
                "retrieved_documents": documents,
                "used_model_knowledge": expand_with_model_knowledge,
                "error": str(e)
            }
