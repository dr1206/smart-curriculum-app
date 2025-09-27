from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import asyncio

from app.db import create_db_and_tables
from app.routers.face import router as face_router
from app.routers.session import router as session_router
from app.notifications import websocket_handler

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

@app.get("/")
def read_root():
    return {"message": "Attendance System API"}

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
