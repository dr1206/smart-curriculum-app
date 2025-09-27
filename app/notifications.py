import asyncio
from typing import Dict
import websockets
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Simple in-memory storage for connected clients
connected_clients: Dict[str, websockets.WebSocketServerProtocol] = {}

def send_email_notification(email: str, subject: str, body: str):
    """
    Send an email notification to the user.
    """
    try:
        # Email configuration (replace with your SMTP settings)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "your_email@gmail.com"  # Replace with your email
        sender_password = "your_password"  # Replace with your password or app password

        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, email, text)
        server.quit()

        print(f"Email sent to {email}")
    except Exception as e:
        print(f"Error sending email to {email}: {e}")

async def notify_user(user_id: str, message: str):
    """
    Send a notification to a specific user via WebSocket.
    """
    if user_id in connected_clients:
        client = connected_clients[user_id]
        try:
            await client.send(json.dumps({"type": "notification", "message": message}))
        except Exception as e:
            print(f"Error sending notification to user {user_id}: {e}")
            # Remove disconnected client
            del connected_clients[user_id]

async def websocket_handler(websocket, path):
    """
    Handle WebSocket connections.
    Expects user_id in query params: ws://localhost:8001?user_id=firebase_uid
    """
    try:
        # Parse user_id from query string
        query_string = path.split('?')[1] if '?' in path else ''
        user_id = None
        if 'user_id=' in query_string:
            user_id = query_string.split('user_id=')[1].split('&')[0]

        if not user_id:
            await websocket.close()
            return

        # Register client
        connected_clients[user_id] = websocket
        print(f"User {user_id} connected")

        # Keep connection alive
        async for message in websocket:
            # Handle incoming messages if needed
            pass

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Unregister client
        if user_id and user_id in connected_clients:
            del connected_clients[user_id]
            print(f"User {user_id} disconnected")
