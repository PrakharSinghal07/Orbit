from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import Conversation, Message
from schemas import NewConversation, ConvPatch
from database import SessionLocal
import uuid

router = APIRouter(prefix="/conversation", tags=["Chatbot"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/initial")
def getInitialConversation(db: Session = Depends(get_db)):
    convo =  db.query(Conversation).order_by(Conversation.id.desc()).first()
    if convo:
        messages = [{"type": msg.type, "text": msg.text} for msg in convo.messages]
        return {
            "sessionId": convo.session_id,
            "title": convo.title,
            "messages": messages
        }
    else:
        session_id = str(uuid.uuid4())
        new_convo = Conversation(session_id=session_id, title="New Chat")
        db.add(new_convo)
        db.commit()
        db.refresh(new_convo)
        
        return {
            "sessionId": session_id,
            "title": "New Chat",
            "messages": []
        }



@router.post("/create")
def createNewConversation(newConversation: NewConversation, db: Session = Depends(get_db)):
    session_id = str(uuid.uuid4())
    convo = Conversation(session_id=session_id, title=newConversation.title)
    db.add(convo)
    db.flush()

    for msg in newConversation.messages:
        new_msg = Message(
            conversation_id=convo.id,
            type=msg.type,
            text=msg.text
        )
        db.add(new_msg)
    
    db.commit()
    return {
        "sessionId": session_id,
        "title": newConversation.title,
        "messages": newConversation.messages
    }



@router.get('/sidebar')
def getSidebarDetails(db: Session = Depends(get_db)):
    convos = db.query(Conversation).all()
    if convos:
        return [{"title": convo.title, "sessionId": convo.session_id} for convo in reversed(convos)]



@router.delete("/{currentSessionID}")
def delete_conversation(currentSessionID: str, db: Session = Depends(get_db)):
    convo = db.query(Conversation).filter(Conversation.session_id == currentSessionID).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(convo)
    db.commit()

    if db.query(Conversation).count() == 0:
        session_id = str(uuid.uuid4())
        new_convo = Conversation(session_id=session_id, title="New Chat")
        db.add(new_convo)
        db.commit()
        db.refresh(new_convo)

    return {"detail": "Conversation deleted"}



@router.post("/{currentSessionID}")
def save_conversation(convo: ConvPatch, currentSessionID: str, db: Session = Depends(get_db)):
    conversation = db.query(Conversation).filter(Conversation.session_id == currentSessionID).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    user_msg = Message(
        conversation_id=conversation.id,
        type=convo.userMsg.type,
        text=convo.userMsg.text
    )
    bot_msg = Message(
        conversation_id=conversation.id,
        type=convo.botMsg.type,
        text=convo.botMsg.text
    )
    db.add(user_msg)
    db.add(bot_msg)

    if conversation.title == "New Chat":
        conversation.title = convo.prompt

    db.commit()
    return {"detail": "Messages saved"}



@router.get("/active/{currentSessionID}")
def getActiveConversation(currentSessionID: str, db: Session = Depends(get_db)):
    convo = db.query(Conversation).filter(Conversation.session_id == currentSessionID).first()
    if convo:
        messages = [{"type": msg.type, "text": msg.text} for msg in convo.messages]
        return {
            "sessionId": convo.session_id,
            "title": convo.title,
            "messages": messages
        }
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")
