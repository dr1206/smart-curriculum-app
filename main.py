from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import asyncio

from app.db import create_db_and_tables
from app.routers.face import router as face_router
from app.routers.session import router as session_router
from app.routers.timetable import router as timetable_router
from app.routers.curriculum import router as curriculum_router
from app.notifications import websocket_handler
from fastapi.staticfiles import StaticFiles

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_db_and_tables()
    yield
    # Shutdown
    pass

app = FastAPI(title="Attendance System API", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(face_router)
app.include_router(session_router)
app.include_router(timetable_router)
app.include_router(curriculum_router)

# Serve static files
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
def read_root():
    return {"message": "Attendance System API"}

@app.get("/code/time.py")
def get_time_code():
    try:
        with open("time.py", "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content)
    except FileNotFoundError:
        return {"error": "File not found"}

if __name__ == "__main__":
    # Start both HTTP and WebSocket servers
    import threading
    import websockets

    def run_websocket():
        async def start_server():
            server = await websockets.serve(websocket_handler, "localhost", 8001)
            await server.wait_closed()
        asyncio.run(start_server())

    # Start WebSocket server in a thread
    ws_thread = threading.Thread(target=run_websocket, daemon=True)
    ws_thread.start()

    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
