# Curriculum Feature Implementation

## Models
- [x] Add Chapter model to app/models.py (id, name, teacher_id, pdf_path, resources)
- [x] Update create_db_and_tables to include new model

## Backend
- [x] Create app/routers/curriculum.py with endpoints:
  - POST /upload-pdf (UploadFile, chapter_name, teacher_id)
  - GET /chapters
  - GET /chapter/{id}/pdf
  - GET /chapter/{id}/resources
- [x] Update main.py to include curriculum router
- [x] Create uploads/pdfs/ folder for storing PDFs

## Frontend
- [x] Update public/teacher.html: Add PDF upload section with file input and chapter name
- [x] Update public/student.html: Add chapters section with list, download links, and resources request

## Testing
- [x] Test upload from teacher (server started)
- [ ] Test view/download from student
- [ ] Test resources request
