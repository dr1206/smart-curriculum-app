from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime
from pydantic import ConfigDict

class User(SQLModel, table=True):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[str] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    role: str  # "student" or "teacher"
    face_embedding: Optional[str] = Field(default=None)
    face_image: Optional[str] = Field(default=None)  # base64 encoded image
    created_at: datetime = Field(default_factory=datetime.utcnow)

class AttendanceSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    teacher_id: str = Field(foreign_key="user.id")
    code: str = Field(unique=True, index=True)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    teacher: Optional[User] = Relationship()

class Attendance(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="attendancesession.id")
    user_id: str = Field(foreign_key="user.id")
    status: str = Field(default="present")  # "present", "absent", etc.
    marked_at: datetime = Field(default_factory=datetime.utcnow)

    session: Optional[AttendanceSession] = Relationship()
    user: Optional[User] = Relationship()

class Chapter(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    teacher_id: str = Field(foreign_key="user.id")
    pdf_path: str
    resources: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    teacher: Optional[User] = Relationship()

class ChapterPlan(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chapter_id: int = Field(foreign_key="chapter.id")
    teacher_id: str = Field(foreign_key="user.id")
    total_lectures: int
    minutes_per_lecture: int
    topics_json: Optional[str] = Field(default=None)
    start_date: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TeachingSchedule(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chapter_id: int = Field(foreign_key="chapter.id")
    teacher_id: str = Field(foreign_key="user.id")
    lecture_number: int
    date: datetime
    topic: str
    is_taught: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)