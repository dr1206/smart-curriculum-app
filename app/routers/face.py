from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Header
from sqlmodel import Session, select
import numpy as np
import json
import base64

from app.db import get_session
from app.models import User, AttendanceSession, Attendance
from app.services.face import compute_face_embedding, cosine_similarity
from app.notifications import notify_user, send_email_notification


router = APIRouter(prefix="/face", tags=["face"]) 


@router.post("/enroll")
async def enroll(user_id: str = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session)):
    data = await file.read()
    emb = compute_face_embedding(data)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected")
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Load existing embeddings
    embeddings = []
    if user.face_embedding:
        try:
            embeddings = json.loads(user.face_embedding)
        except:
            embeddings = []
    # Append new embedding
    embeddings.append(emb.tolist())
    user.face_embedding = json.dumps(embeddings)
    user.face_image = base64.b64encode(data).decode('utf-8')
    session.add(user)
    session.commit()
    return {"message": f"enrolled, total embeddings: {len(embeddings)}"}


@router.post("/verify-and-mark")
async def verify_and_mark(code: str = Form(...), user_id: str = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session)):
    data = await file.read()
    emb = compute_face_embedding(data)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected")
    user = session.get(User, user_id)
    if not user or not user.face_embedding:
        raise HTTPException(status_code=400, detail="User not enrolled")
    embeddings = json.loads(user.face_embedding)
    max_sim = -1.0
    for emb_list in embeddings:
        user_emb = np.array(emb_list, dtype=np.float32)
        sim = cosine_similarity(emb, user_emb)
        if sim > max_sim:
            max_sim = sim
    if max_sim < 0.35:  # conservative threshold for demo
        raise HTTPException(status_code=401, detail="Face mismatch")
    sess = session.exec(select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)).first()  # noqa: E712
    if not sess:
        raise HTTPException(status_code=404, detail="Active session not found")
    existing = session.exec(select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == user_id)).first()
    if existing:
        return {"message": "Already marked present"}
    att = Attendance(session_id=sess.id, user_id=user_id, status="present")
    session.add(att)
    session.commit()
    # Notify student via websocket
    try:
        await notify_user(user_id, f"Attendance marked for session {sess.code}")
    except Exception:
        pass
    return {"message": "Marked present", "similarity": max_sim}


@router.post("/auto-identify-and-mark")
async def auto_identify_and_mark(code: str = Form(...), file: UploadFile = File(...), session: Session = Depends(get_session), x_admin_key: str | None = Header(default=None)):
    import os
    admin_key = os.getenv("ADMIN_KEY")
    if admin_key and (x_admin_key is None or x_admin_key != admin_key):
        raise HTTPException(status_code=403, detail="Forbidden")
    data = await file.read()
    emb = compute_face_embedding(data)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected")
    sess = session.exec(select(AttendanceSession).where(AttendanceSession.code == code, AttendanceSession.is_active == True)).first()  # noqa: E712
    if not sess:
        raise HTTPException(status_code=404, detail="Active session not found")
    # Find best-matching enrolled user
    users = session.exec(select(User).where(User.face_embedding.is_not(None))).all()
    import numpy as np
    best_user = None
    best_sim = -1.0
    for u in users:
        try:
            embeddings = json.loads(u.face_embedding)
            user_max_sim = -1.0
            for emb_list in embeddings:
                uemb = np.array(emb_list, dtype=np.float32)
                s = cosine_similarity(emb, uemb)
                if s > user_max_sim:
                    user_max_sim = s
            if user_max_sim > best_sim:
                best_sim = user_max_sim
                best_user = u
        except Exception:
            continue
    if best_user is None or best_sim < 0.35:
        raise HTTPException(status_code=401, detail="No matching enrolled user")
    best_user_id = best_user.id
    # Mark attendance if not already
    existing = session.exec(select(Attendance).where(Attendance.session_id == sess.id, Attendance.user_id == best_user_id)).first()
    if existing:
        return {"message": "Already marked", "user_id": best_user_id, "similarity": best_sim}
    att = Attendance(session_id=sess.id, user_id=best_user_id, status="present")
    session.add(att)
    session.commit()
    # Notify via WebSocket
    try:
        await notify_user(best_user_id, f"Attendance marked for session {sess.code}")
    except Exception:
        pass
    # Send email notification
    if best_user.email:
        send_email_notification(
            best_user.email,
            "Face Detected - Attendance Marked",
            f"Your face was detected and attendance marked for session {sess.code}. Time: {att.marked_at}"
        )
    return {"message": "Marked present", "user_id": best_user_id, "similarity": best_sim}

@router.get("/attendance")
async def get_attendance(user_id: str, session: Session = Depends(get_session)):
    # Get attendance records for the user
    attendance_records = session.exec(
        select(Attendance, AttendanceSession.code)
        .join(AttendanceSession, Attendance.session_id == AttendanceSession.id)
        .where(Attendance.user_id == user_id)
        .order_by(Attendance.marked_at.desc())
    ).all()

    result = []
    for att, code in attendance_records:
        result.append({
            "session_code": code,
            "marked_at": att.marked_at.isoformat(),
            "status": att.status
        })

    return result

@router.get("/face-image/{user_id}")
async def get_face_image(user_id: str, session: Session = Depends(get_session)):
    user = session.get(User, user_id)
    if not user or not user.face_image:
        raise HTTPException(status_code=404, detail="Face image not found")
    return {"face_image": user.face_image}
