from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from app.db import get_session
from app.models import AttendanceSession

router = APIRouter(prefix="/session", tags=["session"])

@router.post("/start")
def start_session(data: dict, session: Session = Depends(get_session)):
    user_id = data.get('user_id')
    code = data.get('code')
    if not user_id or not code:
        raise HTTPException(status_code=400, detail="user_id and code required")

    # Check if session with code already exists
    existing = session.exec(select(AttendanceSession).where(AttendanceSession.code == code)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Session code already exists")

    new_session = AttendanceSession(code=code, teacher_id=user_id, is_active=True)
    session.add(new_session)
    session.commit()
    session.refresh(new_session)
    return {"message": "Session started", "session_id": new_session.id}

@router.post("/end/{code}")
async def end_session(code: str, session: Session = Depends(get_session)):
    sess = session.exec(select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Active session not found")

    sess.is_active = False
    session.add(sess)
    session.commit()
    return {"message": "Session ended"}

@router.get("/active/{teacher_id}")
async def get_active_session(teacher_id: str, session: Session = Depends(get_session)):
    sess = session.exec(select(AttendanceSession).where(AttendanceSession.teacher_id == teacher_id, AttendanceSession.is_active == True)).first()
    if not sess:
        return None
    return {"code": sess.code, "id": sess.id}
