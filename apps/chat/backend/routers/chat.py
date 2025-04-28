from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from pydantic import BaseModel
from services.llm_service import get_llm_response

router = APIRouter(prefix="/chat", tags=["Chatbot"])

class ChatResponse(BaseModel):
    response: str

@router.post("/", response_model=ChatResponse)
async def get_chat_response(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    print("Received message:", message)
    if file:
        print("Received file:", file.filename)

    response = get_llm_response(message)
    return {"response": response}
