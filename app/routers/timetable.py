from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
import datetime
import pytz
from sqlmodel import Session, select
from app.db import get_session
from app.models import ChapterPlan, TeachingSchedule

router = APIRouter(prefix="/timetable", tags=["timetable"])

class SubjectDifficulty(BaseModel):
    name: str
    difficulty: int

class TimetableRequest(BaseModel):
    subjects: List[SubjectDifficulty]
    busy_hours: Dict[str, str]  # day: "9-12,14-16"
    max_blocks_per_day: int = 4
    timezone: str = "UTC"

class PomodoroTimetableGenerator:
    def __init__(self):
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.study_start_hour = 6
        self.study_end_hour = 21
        self.pomodoro_block_duration = 1.25

    def parse_time_ranges(self, time_input: str) -> List[float]:
        busy_times = []
        if not time_input.strip():
            return busy_times

        try:
            ranges = time_input.split(',')
            for range_str in ranges:
                range_str = range_str.strip()
                if '-' in range_str:
                    start, end = range_str.split('-')
                    start_hour = float(start.strip())
                    end_hour = float(end.strip())

                    current = start_hour
                    while current < end_hour:
                        busy_times.append(current)
                        current += 0.25
                else:
                    busy_times.append(float(range_str))
        except ValueError:
            return []

        return busy_times

    def get_available_pomodoro_slots(self, busy_hours: Dict[str, List[float]]) -> Dict[str, List[float]]:
        available_slots = {}

        for day in self.days:
            busy_today = set(busy_hours.get(day, []))
            available_blocks = []

            current_time = self.study_start_hour
            while current_time + self.pomodoro_block_duration <= self.study_end_hour:
                block_free = True
                check_time = current_time

                while check_time < current_time + self.pomodoro_block_duration:
                    if check_time in busy_today:
                        block_free = False
                        break
                    check_time += 0.25

                if block_free:
                    available_blocks.append(current_time)

                current_time += self.pomodoro_block_duration

            available_slots[day] = available_blocks

        return available_slots

    def calculate_subject_distribution(self, subjects: Dict[str, int], total_blocks_per_week: int) -> Dict[str, int]:
        if not subjects:
            return {}

        total_difficulty = sum(subjects.values())
        min_blocks_per_subject = 1
        base_blocks = len(subjects) * min_blocks_per_subject
        remaining_blocks = max(0, total_blocks_per_week - base_blocks)

        allocation = {}
        for subject, difficulty in subjects.items():
            base_allocation = min_blocks_per_subject
            difficulty_bonus = int((difficulty / total_difficulty) * remaining_blocks)
            allocation[subject] = base_allocation + difficulty_bonus

        total_allocated = sum(allocation.values())
        if total_allocated > total_blocks_per_week:
            reduction_factor = total_blocks_per_week / total_allocated
            allocation = {subject: max(1, int(blocks * reduction_factor))
                         for subject, blocks in allocation.items()}

        return allocation

    def create_pomodoro_timetable(self, subjects: Dict[str, int], busy_hours: Dict[str, List[float]],
                                 max_blocks_per_day: int) -> Dict[str, List[Dict]]:
        available_slots = self.get_available_pomodoro_slots(busy_hours)
        total_available_blocks = sum(min(len(slots), max_blocks_per_day)
                                   for slots in available_slots.values())

        subject_allocation = self.calculate_subject_distribution(subjects, total_available_blocks)

        timetable = {}
        for day in self.days:
            timetable[day] = []

        assigned_blocks = {subject: 0 for subject in subjects.keys()}
        sorted_subjects = sorted(subjects.items(), key=lambda x: x[1], reverse=True)

        for day in self.days:
            day_slots = available_slots[day][:max_blocks_per_day]

            available_subjects = []
            for subject, difficulty in sorted_subjects:
                needed_blocks = subject_allocation[subject]
                if assigned_blocks[subject] < needed_blocks:
                    blocks_needed = needed_blocks - assigned_blocks[subject]
                    available_subjects.extend([subject] * min(blocks_needed, 3))

            for i, slot_start in enumerate(day_slots):
                if i < len(available_subjects):
                    subject = available_subjects[i % len(available_subjects)]

                    study_end = slot_start + 1
                    break_end = study_end + 0.25

                    pomodoro_block = {
                        'subject': subject,
                        'study_start': slot_start,
                        'study_end': study_end,
                        'break_end': break_end,
                        'difficulty': subjects[subject]
                    }

                    timetable[day].append(pomodoro_block)
                    assigned_blocks[subject] += 1

        return timetable

@router.post("/generate")
async def generate_timetable(request: TimetableRequest):
    generator = PomodoroTimetableGenerator()

    subjects_dict = {s.name: s.difficulty for s in request.subjects}
    busy_hours_parsed = {day: generator.parse_time_ranges(times) for day, times in request.busy_hours.items()}

    timetable = generator.create_pomodoro_timetable(
        subjects_dict, busy_hours_parsed, request.max_blocks_per_day
    )

    return {"timetable": timetable}


class ChapterPlanRequest(BaseModel):
    chapter_id: int
    teacher_id: str
    total_lectures: int
    minutes_per_lecture: int
    topics: Optional[List[str]] = None
    start_date: Optional[str] = None  # YYYY-MM-DD


@router.post("/plan")
def create_chapter_plan(data: ChapterPlanRequest, session: Session = Depends(get_session)):
    existing = session.exec(
        select(ChapterPlan).where(
            (ChapterPlan.chapter_id == data.chapter_id) & (ChapterPlan.teacher_id == data.teacher_id)
        )
    ).first()
    if existing:
        existing.total_lectures = data.total_lectures
        existing.minutes_per_lecture = data.minutes_per_lecture
        existing.topics_json = ",".join(data.topics) if data.topics else None
        existing.start_date = datetime.datetime.strptime(data.start_date, "%Y-%m-%d") if data.start_date else None
        session.add(existing)
        session.commit()
        session.refresh(existing)
        plan = existing
    else:
        plan = ChapterPlan(
            chapter_id=data.chapter_id,
            teacher_id=data.teacher_id,
            total_lectures=data.total_lectures,
            minutes_per_lecture=data.minutes_per_lecture,
            topics_json=",".join(data.topics) if data.topics else None,
            start_date=datetime.datetime.strptime(data.start_date, "%Y-%m-%d") if data.start_date else None,
        )
        session.add(plan)
        session.commit()
        session.refresh(plan)

    return {"message": "Plan saved", "plan_id": plan.id}


class ScheduleGenerateRequest(BaseModel):
    chapter_id: int
    teacher_id: str
    start_date: str  # YYYY-MM-DD


@router.post("/schedule/generate")
def generate_and_store_schedule(data: ScheduleGenerateRequest, session: Session = Depends(get_session)):
    plan = session.exec(
        select(ChapterPlan).where(
            (ChapterPlan.chapter_id == data.chapter_id) & (ChapterPlan.teacher_id == data.teacher_id)
        )
    ).first()
    if not plan:
        raise HTTPException(status_code=404, detail="No plan found for this chapter and teacher")

    # Clear any existing schedule for this chapter/teacher
    existing = session.exec(
        select(TeachingSchedule).where(
            (TeachingSchedule.chapter_id == data.chapter_id) & (TeachingSchedule.teacher_id == data.teacher_id)
        )
    ).all()
    for row in existing:
        session.delete(row)
    session.commit()

    topics: List[str] = plan.topics_json.split(",") if plan.topics_json else []
    total_lectures = plan.total_lectures
    if topics and len(topics) != total_lectures:
        # If topics count doesn't match, distribute as best-effort
        while len(topics) < total_lectures:
            topics.append(f"Lecture {len(topics)+1}")
        topics = topics[:total_lectures]
    elif not topics:
        topics = [f"Lecture {i+1}" for i in range(total_lectures)]

    start_date = datetime.datetime.strptime(data.start_date, "%Y-%m-%d").date()

    created: List[TeachingSchedule] = []
    current_date = start_date
    for idx in range(total_lectures):
        # Skip Sundays for teaching by default
        while current_date.weekday() == 6:
            current_date += datetime.timedelta(days=1)

        schedule = TeachingSchedule(
            chapter_id=data.chapter_id,
            teacher_id=data.teacher_id,
            lecture_number=idx + 1,
            date=datetime.datetime.combine(current_date, datetime.time(9, 0)),
            topic=topics[idx],
        )
        session.add(schedule)
        session.commit()
        session.refresh(schedule)
        created.append(schedule)

        current_date += datetime.timedelta(days=1)

    return {"message": "Schedule generated", "count": len(created)}


@router.get("/schedule/{teacher_id}/{chapter_id}")
def get_schedule(teacher_id: str, chapter_id: int, session: Session = Depends(get_session)):
    rows = session.exec(
        select(TeachingSchedule).where(
            (TeachingSchedule.chapter_id == chapter_id) & (TeachingSchedule.teacher_id == teacher_id)
        ).order_by(TeachingSchedule.lecture_number)
    ).all()
    return [
        {
            "id": r.id,
            "lecture_number": r.lecture_number,
            "date": r.date.strftime("%Y-%m-%d"),
            "topic": r.topic,
            "is_taught": r.is_taught,
        }
        for r in rows
    ]


class MarkTaughtRequest(BaseModel):
    schedule_id: int
    is_taught: bool = True


@router.post("/schedule/mark-taught")
def mark_taught(data: MarkTaughtRequest, session: Session = Depends(get_session)):
    row = session.get(TeachingSchedule, data.schedule_id)
    if not row:
        raise HTTPException(status_code=404, detail="Schedule row not found")
    row.is_taught = data.is_taught
    session.add(row)
    session.commit()
    session.refresh(row)
    return {"message": "Updated", "id": row.id, "is_taught": row.is_taught}
