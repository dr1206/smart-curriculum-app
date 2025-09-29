from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlmodel import Session, select
from app.db import get_session
from app.models import Chapter
import os
import shutil
from fastapi.responses import FileResponse

router = APIRouter(prefix="/curriculum", tags=["curriculum"])

UPLOAD_DIR = "uploads/pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    chapter_name: str = Form(...),
    teacher_id: str = Form(...),
    session: Session = Depends(get_session)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Save file
    file_path = os.path.join(UPLOAD_DIR, f"{teacher_id}_{chapter_name.replace(' ', '_')}.pdf")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create chapter
    chapter = Chapter(name=chapter_name, teacher_id=teacher_id, pdf_path=file_path)
    session.add(chapter)
    session.commit()
    session.refresh(chapter)
    return {"message": "PDF uploaded successfully", "chapter_id": chapter.id}

@router.get("/chapters")
def get_chapters(session: Session = Depends(get_session)):
    chapters = session.exec(select(Chapter)).all()
    return [{"id": c.id, "name": c.name, "teacher_id": c.teacher_id} for c in chapters]

@router.get("/chapter/{chapter_id}/pdf")
def download_pdf(chapter_id: int, session: Session = Depends(get_session)):
    chapter = session.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    if not os.path.exists(chapter.pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    return FileResponse(chapter.pdf_path, media_type='application/pdf', filename=f"{chapter.name}.pdf")

@router.get("/chapter/{chapter_id}/resources")
def get_resources(chapter_id: int, session: Session = Depends(get_session)):
    chapter = session.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    # For demo, return sample resources if none
    resources = chapter.resources or f"Sample resources for {chapter.name}: Books, Videos, Quizzes."
    return {"resources": resources}
