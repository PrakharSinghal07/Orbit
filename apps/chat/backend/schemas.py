from pydantic import BaseModel, Field
from typing import List, Literal


class ChatRequest(BaseModel):
  query: str


class ChatResponse(BaseModel):
  response: str


class Suggestions(BaseModel):
  Suggestions: List[str]
  
class Message(BaseModel):
    type: Literal["user", "bot"]
    text: str

    
class NewConversation(BaseModel):
    title: str
    messages: List[Message]
    

class ConvPatch(BaseModel):
  userMsg: Message
  botMsg: Message
  prompt: str