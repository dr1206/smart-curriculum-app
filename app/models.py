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
