from fastapi import APIRouter, Depends, Query, HTTPException, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from services.search_service import SearchService
from config import settings

router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={404: {"description": "Not found"}},
)

class SearchResult(BaseModel):
    id: Any
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchRequest(BaseModel):
    query: str
    collection_name: Optional[str] = settings.DEFAULT_COLLECTION_NAME
    limit: Optional[int] = 5
    gemini_api_key: Optional[str] = None
    rerank: Optional[bool] = True
    filter_conditions: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    count: int

@router.post("/", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    search_service: SearchService = Depends(lambda: SearchService())
):
    """
    Search for documents in the vector database.
    """
    results = search_service.search(
        query_text=request.query,
        collection_name=request.collection_name,
        limit=request.limit,
        gemini_api_key=request.gemini_api_key,
        rerank=request.rerank,
        filter_conditions=request.filter_conditions
    )
    
    formatted_results = []
    for result in results:
        metadata = {k: v for k, v in result.payload.items() if k != 'text'}
        formatted_results.append(
            SearchResult(
                id=result.id,
                score=result.score,
                text=result.payload.get('text', ''),
                metadata=metadata
            )
        )
    
    return {
        "query": request.query,
        "results": formatted_results,
        "count": len(formatted_results)
    }