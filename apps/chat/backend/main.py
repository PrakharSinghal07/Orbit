from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat, suggestions, conversation
import os
from dotenv import load_dotenv  
from database import Base, engine


load_dotenv()
frontendURL = os.getenv("FRONTEND_URL")


app = FastAPI()


origins = [
    frontendURL,
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


Base.metadata.create_all(bind=engine)


app.include_router(chat.router)
app.include_router(suggestions.router)
app.include_router(conversation.router)
