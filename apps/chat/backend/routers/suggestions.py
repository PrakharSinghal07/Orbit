from fastapi import APIRouter
from schemas import Suggestions
from services.llm_service import getSuggestions
router = APIRouter(prefix="/suggestions", tags=["Chatbot"])

@router.get("/")
def get_chat_response():
  suggestions = getSuggestions()
  return {'suggestions': suggestions}