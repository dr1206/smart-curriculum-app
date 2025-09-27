import streamlit as st
import pandas as pd
import datetime
import json
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
import math
import os
import pickle
import pytz
from datetime import timedelta

# Google Calendar imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False
    st.warning("Google Calendar integration not available. Install required packages: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

class GoogleCalendarIntegrator:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/calendar']
        self.credentials = None
        self.service = None
        
    def get_client_config(self):
        """Get client configuration from secrets or environment variables"""
        try:
            client_config = {
                "web": {
                    "client_id": 1037125480486- "3m09cd56ug9dj91bvjq7a5f1gdl2qmip.apps.googleusercontent.com",
                    "client_secret": "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",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost:8501"]
                }
            }
            
            if not client_config["web"]["client_id"] or not client_config["web"]["client_secret"]:
                return None
                
            return client_config
        except Exception:
            return None
    
    def is_authenticated(self):
        """Check if user is already authenticated"""
        if 'google_credentials' in st.session_state:
            creds = st.session_state.google_credentials
            if creds and creds.valid:
                self.credentials = creds
                self.service = build('calendar', 'v3', credentials=creds)
                return True
            elif creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    st.session_state.google_credentials = creds
                    self.credentials = creds
                    self.service = build('calendar', 'v3', credentials=creds)
                    return True
                except Exception:
                    return False
        return False
    
    def get_auth_url(self):
        """Get authorization URL for OAuth flow"""
        client_config = self.get_client_config()
        if not client_config:
            return None
            
        try:
            flow = Flow.from_client_config(client_config, self.SCOPES)
            flow.redirect_uri = "http://localhost:8501"
            
            auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
            st.session_state.oauth_flow = flow
            
            return auth_url
        except Exception as e:
            st.error(f"Error generating auth URL: {e}")
            return None
    
    def handle_auth_code(self, auth_code):
        """Handle the authorization code from OAuth flow"""
        try:
            if 'oauth_flow' not in st.session_state:
                return False
                
            flow = st.session_state.oauth_flow
            flow.fetch_token(code=auth_code)
            
            st.session_state.google_credentials = flow.credentials
            self.credentials = flow.credentials
            self.service = build('calendar', 'v3', credentials=flow.credentials)
            
            return True
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            return False
    
    def create_calendar_event(self, subject, start_datetime, end_datetime, description="", timezone='America/New_York'):
        """Create a single calendar event"""
        if not self.service:
            return None
        
        event = {
            'summary': f'🍅 Study: {subject}',
            'description': description,
            'start': {
                'dateTime': start_datetime.isoformat(),
                'timeZone': timezone,
            },
            'end': {
                'dateTime': end_datetime.isoformat(),
                'timeZone': timezone,
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'popup', 'minutes': 10},       # 10 minutes before
                    {'method': 'popup', 'minutes': 60},       # 1 hour before
                ],
            },
            'colorId': '7',  # Peacock blue for study sessions
        }
        
        try:
            created_event = self.service.events().insert(calendarId='primary', body=event).execute()
            return created_event.get('htmlLink')
        except Exception as e:
            st.error(f"Error creating calendar event: {e}")
            return None
    
    def create_study_schedule_events(self, timetable, timezone='America/New_York', start_date=None):
        """Create calendar events for the entire study schedule"""
        if not self.service:
            return []
        
        if start_date is None:
            start_date = datetime.datetime.now().date()
        
        created_events = []
        days_mapping = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        
        for day_name, blocks in timetable.items():
            if not blocks:  # Skip days with no blocks
                continue
                
            # Calculate the date for this day of week
            days_ahead = days_mapping[day_name]
            days_until_target = (days_ahead - start_date.weekday()) % 7
            target_date = start_date + datetime.timedelta(days=days_until_target)
            
            for block in blocks:
                # Create study event
                study_start_hour = int(block['study_start'])
                study_start_minute = int((block['study_start'] % 1) * 60)
                study_end_hour = int(block['study_end'])
                study_end_minute = int((block['study_end'] % 1) * 60)
                
                study_start_datetime = datetime.datetime.combine(
                    target_date, 
                    datetime.time(hour=study_start_hour, minute=study_start_minute)
                )
                study_end_datetime = datetime.datetime.combine(
                    target_date, 
                    datetime.time(hour=study_end_hour, minute=study_end_minute)
                )
                
                # Localize to timezone
                tz = pytz.timezone(timezone)
                study_start_datetime = tz.localize(study_start_datetime)
                study_end_datetime = tz.localize(study_end_datetime)
                
                description = f"""🍅 Pomodoro Study Session

📚 Subject: {block['subject']}
⭐ Difficulty: {block['difficulty']}/10
⏰ Duration: 60 minutes

💡 Tips:
• Remove all distractions
• Focus on this subject only
• Take notes actively
• Stay hydrated

After this session, take a 15-minute break! 🎯"""
                
                event_link = self.create_calendar_event(
                    block['subject'],
                    study_start_datetime,
                    study_end_datetime,
                    description,
                    timezone
                )
                
                if event_link:
                    created_events.append({
                        'subject': block['subject'],
                        'day': day_name,
                        'date': target_date.strftime('%Y-%m-%d'),
                        'time': f"{study_start_hour:02d}:{study_start_minute:02d}",
                        'link': event_link
                    })
        
        return created_events
    
    def get_user_calendars(self):
        """Get list of user's calendars"""
        if not self.service:
            return []
        
        try:
            calendar_list = self.service.calendarList().list().execute()
            calendars = []
            for calendar in calendar_list.get('items', []):
                calendars.append({
                    'id': calendar['id'],
                    'summary': calendar['summary'],
                    'primary': calendar.get('primary', False)
                })
            return calendars
        except Exception as e:
            st.error(f"Error fetching calendars: {e}")
            return []

class PomodoroTimetableGenerator:
    def __init__(self):
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Study blocks from 6 AM to 9 PM (15 hours available)
        # Each block is 1 hour study + 15 min break = 1.25 hours
        self.study_start_hour = 6  # 6 AM
        self.study_end_hour = 21   # 9 PM (21:00)
        self.pomodoro_block_duration = 1.25  # 1 hour study + 15 min break
        
    def format_time(self, hour: int, minute: int = 0) -> str:
        """Convert 24-hour format to readable time"""
        total_minutes = hour * 60 + minute
        display_hour = total_minutes // 60
        display_minute = total_minutes % 60
        
        if display_hour == 0:
            return f"12:{display_minute:02d} AM"
        elif display_hour < 12:
            return f"{display_hour}:{display_minute:02d} AM"
        elif display_hour == 12:
            return f"12:{display_minute:02d} PM"
        else:
            return f"{display_hour-12}:{display_minute:02d} PM"
    
    def parse_time_ranges(self, time_input: str) -> List[float]:
        """Parse time ranges and convert to decimal hours"""
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
                    
                    # Add all 15-minute intervals in the range
                    current = start_hour
                    while current < end_hour:
                        busy_times.append(current)
                        current += 0.25  # 15-minute increments
                else:
                    # Single hour
                    busy_times.append(float(range_str))
        except ValueError:
            st.error(f"Invalid time format: {time_input}. Use format like '9-12, 14-16'")
            return []
        
        return busy_times
    
    def get_available_pomodoro_slots(self, busy_hours: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Find available Pomodoro time slots (1.25-hour blocks) for each day"""
        available_slots = {}
        
        for day in self.days:
            busy_today = set(busy_hours.get(day, []))
            available_blocks = []
            
            # Check each possible Pomodoro block start time
            current_time = self.study_start_hour
            while current_time + self.pomodoro_block_duration <= self.study_end_hour:
                # Check if this entire 1.25-hour block is free
                block_free = True
                check_time = current_time
                
                while check_time < current_time + self.pomodoro_block_duration:
                    if check_time in busy_today:
                        block_free = False
                        break
                    check_time += 0.25  # Check every 15 minutes
                
                if block_free:
                    available_blocks.append(current_time)
                
                current_time += self.pomodoro_block_duration  # Move to next possible block
            
            available_slots[day] = available_blocks
        
        return available_slots
    
    def calculate_subject_distribution(self, subjects: Dict[str, int], total_blocks_per_week: int) -> Dict[str, int]:
        """
        Distribute subjects evenly throughout the week based on difficulty
        """
        if not subjects:
            return {}
        
        # Calculate total difficulty points
        total_difficulty = sum(subjects.values())
        
        # Ensure each subject gets at least 1 block per week
        min_blocks_per_subject = 1
        base_blocks = len(subjects) * min_blocks_per_subject
        remaining_blocks = max(0, total_blocks_per_week - base_blocks)
        
        # Distribute remaining blocks based on difficulty
        allocation = {}
        for subject, difficulty in subjects.items():
            base_allocation = min_blocks_per_subject
            difficulty_bonus = int((difficulty / total_difficulty) * remaining_blocks)
            allocation[subject] = base_allocation + difficulty_bonus
        
        # Ensure we don't exceed total blocks
        total_allocated = sum(allocation.values())
        if total_allocated > total_blocks_per_week:
            # Reduce allocations proportionally
            reduction_factor = total_blocks_per_week / total_allocated
            allocation = {subject: max(1, int(blocks * reduction_factor)) 
                         for subject, blocks in allocation.items()}
        
        return allocation
    
    def create_pomodoro_timetable(self, subjects: Dict[str, int], busy_hours: Dict[str, List[float]], 
                                 max_blocks_per_day: int) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
        """
        Create Pomodoro-based timetable with even distribution
        """
        # Get available Pomodoro slots
        available_slots = self.get_available_pomodoro_slots(busy_hours)
        
        # Calculate total available blocks per week
        total_available_blocks = sum(min(len(slots), max_blocks_per_day) 
                                   for slots in available_slots.values())
        
        # Distribute subjects based on difficulty
        subject_allocation = self.calculate_subject_distribution(subjects, total_available_blocks)
        
        # Create timetable structure
        timetable = {}
        for day in self.days:
            timetable[day] = []
        
        # Track how many blocks each subject has been assigned
        assigned_blocks = {subject: 0 for subject in subjects.keys()}
        
        # Sort subjects by difficulty (hardest first) for better time slots
        sorted_subjects = sorted(subjects.items(), key=lambda x: x[1], reverse=True)
        
        # Assign subjects to time slots with even distribution
        for day in self.days:
            day_slots = available_slots[day][:max_blocks_per_day]
            
            # Create a rotation of subjects for even distribution
            available_subjects = []
            for subject, difficulty in sorted_subjects:
                needed_blocks = subject_allocation[subject]
                if assigned_blocks[subject] < needed_blocks:
                    # Add subject multiple times based on how many blocks it needs
                    blocks_needed = needed_blocks - assigned_blocks[subject]
                    available_subjects.extend([subject] * min(blocks_needed, 3))  # Max 3 per day
            
            # Assign subjects to available slots for this day
            for i, slot_start in enumerate(day_slots):
                if i < len(available_subjects):
                    subject = available_subjects[i % len(available_subjects)]
                    
                    # Create the Pomodoro block
                    study_end = slot_start + 1  # 1 hour study
                    break_end = study_end + 0.25  # 15 minute break
                    
                    pomodoro_block = {
                        'subject': subject,
                        'study_start': slot_start,
                        'study_end': study_end,
                        'break_end': break_end,
                        'difficulty': subjects[subject]
                    }
                    
                    timetable[day].append(pomodoro_block)
                    assigned_blocks[subject] += 1
        
        return timetable, subject_allocation
    
    def create_schedule_visualization(self, timetable: Dict[str, List[Dict]], subjects: Dict[str, int]):
        """Create a detailed schedule visualization"""
        
        # Prepare data for the timeline chart
        schedule_data = []
        colors = px.colors.qualitative.Set3
        subject_colors = {subject: colors[i % len(colors)] for i, subject in enumerate(subjects.keys())}
        
        for day_idx, day in enumerate(self.days):
            day_schedule = timetable[day]
            
            for block in day_schedule:
                # Study block
                schedule_data.append({
                    'Day': day,
                    'Start': block['study_start'],
                    'End': block['study_end'],
                    'Subject': block['subject'],
                    'Type': 'Study',
                    'Color': subject_colors[block['subject']],
                    'Day_Num': day_idx
                })
                
                # Break block
                schedule_data.append({
                    'Day': day,
                    'Start': block['study_end'],
                    'End': block['break_end'],
                    'Subject': 'Break',
                    'Type': 'Break',
                    'Color': '#E8E8E8',
                    'Day_Num': day_idx
                })
        
        if not schedule_data:
            return None
            
        df = pd.DataFrame(schedule_data)
        
        # Create Gantt-style chart
        fig = go.Figure()
        
        for _, row in df.iterrows():
            fig.add_trace(go.Bar(
                name=row['Subject'],
                y=[row['Day']],
                x=[row['End'] - row['Start']],
                base=[row['Start']],
                orientation='h',
                marker_color=row['Color'],
                text=f"{row['Subject']}" if row['Type'] == 'Study' else 'Break',
                textposition='inside',
                showlegend=False,
                hovertemplate=f"<b>{row['Subject']}</b><br>" +
                             f"Time: {self.format_time(int(row['Start']), int((row['Start'] % 1) * 60))}-" +
                             f"{self.format_time(int(row['End']), int((row['End'] % 1) * 60))}<br>" +
                             f"Duration: {(row['End'] - row['Start']) * 60:.0f} minutes<extra></extra>"
            ))
        
        fig.update_layout(
            title="Weekly Pomodoro Study Schedule",
            xaxis_title="Time of Day",
            yaxis_title="Day",
            height=600,
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(6, 22, 2)),
                ticktext=[self.format_time(h) for h in range(6, 22, 2)],
                range=[6, 21]
            ),
            barmode='stack'
        )
        
        return fig
    
    def create_daily_summary(self, timetable: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Create a summary table of daily schedules"""
        summary_data = []
        
        for day in self.days:
            day_blocks = timetable[day]
            if day_blocks:
                study_subjects = [block['subject'] for block in day_blocks]
                total_study_time = len(day_blocks)  # Each block is 1 hour
                start_time = min(block['study_start'] for block in day_blocks)
                end_time = max(block['break_end'] for block in day_blocks)
                
                summary_data.append({
                    'Day': day,
                    'Study Hours': total_study_time,
                    'Subjects': ', '.join(study_subjects),
                    'Start Time': self.format_time(int(start_time), int((start_time % 1) * 60)),
                    'End Time': self.format_time(int(end_time), int((end_time % 1) * 60))
                })
            else:
                summary_data.append({
                    'Day': day,
                    'Study Hours': 0,
                    'Subjects': 'Rest Day',
                    'Start Time': '-',
                    'End Time': '-'
                })
        
        return pd.DataFrame(summary_data)

def main():
    st.set_page_config(
        page_title="Pomodoro Study Timetable Generator",
        page_icon="🍅",
        layout="wide"
    )
    
    st.title("🍅 Pomodoro Study Timetable Generator")
    st.markdown("### Create your optimized study schedule with 1-hour focus blocks + 15-minute breaks!")
    
    generator = PomodoroTimetableGenerator()
    
    # Initialize Google Calendar integrator if available
    calendar_integrator = None
    if GOOGLE_CALENDAR_AVAILABLE:
        calendar_integrator = GoogleCalendarIntegrator()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📝 Setup Your Schedule")
        
        # Study preferences
        st.subheader("⏰ Study Preferences")
        max_blocks_per_day = st.slider(
            "Maximum study blocks per day:",
            min_value=1,
            max_value=8,
            value=4,
            help="Each block = 1 hour study + 15 min break"
        )
        
        total_daily_time = max_blocks_per_day * 1.25
        st.info(f"📊 Daily time commitment: {total_daily_time:.1f} hours\n(Including breaks, ending by 9 PM)")
        
        # Subjects and difficulty
        st.subheader("📚 Subjects & Difficulty")
        st.markdown("*Rate difficulty from 1 (easiest) to 10 (hardest)*")
        
        subjects = {}
        num_subjects = st.number_input("Number of subjects:", min_value=1, max_value=8, value=4)
        
        for i in range(num_subjects):
            col1, col2 = st.columns([2, 1])
            with col1:
                subject_name = st.text_input(f"Subject {i+1}:", key=f"subject_{i}")
            with col2:
                if subject_name:
                    difficulty = st.selectbox(
                        f"Difficulty:",
                        options=list(range(1, 11)),
                        index=4,
                        key=f"difficulty_{i}"
                    )
                    subjects[subject_name] = difficulty
        
        # Busy hours input
        st.subheader("🚫 Busy Hours (6 AM - 9 PM)")
        st.markdown("*Enter when you're NOT available for study blocks*")
        st.markdown("**Format:** `9-12, 14-16` (busy 9 AM-12 PM, 2-4 PM)")
        
        busy_hours = {}
        for day in generator.days:
            time_input = st.text_input(
                f"{day}:",
                placeholder="e.g., 8-12, 14-17",
                key=f"busy_{day}",
                help="Leave empty if available all day"
            )
            busy_hours[day] = generator.parse_time_ranges(time_input)
        
        # Google Calendar Integration
        enable_calendar = False
        timezone = 'America/New_York'
        start_date = datetime.datetime.now().date()
        
        if GOOGLE_CALENDAR_AVAILABLE and calendar_integrator:
            st.subheader("📅 Google Calendar Integration")
            enable_calendar = st.checkbox("Add events to Google Calendar", value=False)
            
            if enable_calendar:
                timezone = st.selectbox(
                    "Select your timezone:",
                    options=[
                        'America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles',
                        'Europe/London', 'Europe/Paris', 'Europe/Berlin', 'Europe/Rome',
                        'Asia/Tokyo', 'Asia/Shanghai', 'Asia/Kolkata', 'Asia/Dubai',
                        'Australia/Sydney', 'Australia/Melbourne'
                    ],
                    index=0
                )
                
                start_date = st.date_input(
                    "Start date for schedule:",
                    value=datetime.datetime.now().date(),
                    min_value=datetime.datetime.now().date()
                )
                
                # Check authentication status
                if calendar_integrator.is_authenticated():
                    st.success("✅ Connected to Google Calendar")
                else:
                    st.info("🔐 Google Calendar authentication required")
    
    # Main content
    if st.sidebar.button("🚀 Generate Pomodoro Schedule!", type="primary"):
        if not subjects:
            st.error("Please enter at least one subject!")
            return
        
        with st.spinner("Creating your Pomodoro study schedule..."):
            timetable, subject_allocation = generator.create_pomodoro_timetable(
                subjects, busy_hours, max_blocks_per_day
            )
        
        st.success("✅ Your Pomodoro study schedule is ready!")
        
        # Handle Google Calendar integration
        calendar_events = []
        if enable_calendar and GOOGLE_CALENDAR_AVAILABLE and calendar_integrator:
            if not calendar_integrator.is_authenticated():
                # Show authentication flow
                st.subheader("📅 Google Calendar Authentication")
                
                if calendar_integrator.get_client_config():
                    auth_url = calendar_integrator.get_auth_url()
                    
                    if auth_url:
                        st.markdown("**Step 1:** Click the link below to authorize Google Calendar access:")
                        st.markdown(f"[🔐 Authorize Google Calendar Access]({auth_url})")
                        
                        st.markdown("**Step 2:** Copy the authorization code and paste it below:")
                        auth_code = st.text_input("Authorization code:", key="auth_code")
                        
                        if st.button("Connect to Google Calendar") and auth_code:
                            if calendar_integrator.handle_auth_code(auth_code):
                                st.success("✅ Successfully connected to Google Calendar!")
                                st.rerun()
                            else:
                                st.error("❌ Authentication failed. Please try again.")
                    else:
                        st.error("❌ Could not generate authorization URL. Please check your credentials.")
                else:
                    st.error("❌ Google Calendar credentials not configured. Please set up GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.")
            else:
                # Create calendar events
                with st.spinner("Adding events to Google Calendar..."):
                    calendar_events = calendar_integrator.create_study_schedule_events(
                        timetable, timezone, start_date
                    )
                
                if calendar_events:
                    st.success(f"📅 Successfully created {len(calendar_events)} calendar events!")
                    
                    with st.expander("📅 Created Calendar Events"):
                        for event in calendar_events:
                            st.markdown(f"• **{event['subject']}** on {event['day']} ({event['date']}) at {event['time']}")
                else:
                    st.warning("⚠️ No calendar events were created.")
        
        # Display key metrics
        total_blocks = sum(len(blocks) for blocks in timetable.values())
        total_study_hours = total_blocks
        total_time_with_breaks = total_blocks * 1.25
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Weekly Study Blocks", total_blocks)
        with col2:
            st.metric("Pure Study Hours", f"{total_study_hours}h")
        with col3:
            st.metric("Total Time (with breaks)", f"{total_time_with_breaks:.1f}h")
        with col4:
            st.metric("Daily Average", f"{total_time_with_breaks/7:.1f}h")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📅 Visual Schedule", "📋 Daily Breakdown", "📊 Analytics", "💾 Export"])
        
        with tab1:
            st.subheader("Your Pomodoro Schedule Timeline")
            fig = generator.create_schedule_visualization(timetable, subjects)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Pomodoro explanation
            with st.expander("🍅 How Pomodoro Technique Works"):
                st.markdown("""
                **Each Study Block Contains:**
                - 🎯 **60 minutes** of focused study
                - ☕ **15 minutes** break for rest/refresh
                - 🔄 **Total: 1 hour 15 minutes** per block
                
                **Benefits:**
                - Maintains high concentration for optimal learning
                - Prevents mental fatigue
                - Builds sustainable study habits
                - Better retention and understanding
                """)
        
        with tab2:
            st.subheader("Daily Schedule Breakdown")
            summary_df = generator.create_daily_summary(timetable)
            st.dataframe(summary_df, use_container_width=True)
            
            # Detailed daily view
            st.subheader("Detailed Daily Schedules")
            for day in generator.days:
                day_blocks = timetable[day]
                if day_blocks:
                    with st.expander(f"📅 {day} ({len(day_blocks)} blocks)"):
                        for i, block in enumerate(day_blocks, 1):
                            study_start_str = generator.format_time(int(block['study_start']), 
                                                                  int((block['study_start'] % 1) * 60))
                            study_end_str = generator.format_time(int(block['study_end']), 
                                                                int((block['study_end'] % 1) * 60))
                            break_end_str = generator.format_time(int(block['break_end']), 
                                                                int((block['break_end'] % 1) * 60))
                            
                            st.markdown(f"""
                            **Block {i}: {block['subject']}** (Difficulty: {block['difficulty']}/10)
                            - 🎯 Study: {study_start_str} - {study_end_str}
                            - ☕ Break: {study_end_str} - {break_end_str}
                            """)
        
        with tab3:
            st.subheader("Study Analytics")
            
            # Subject allocation
            allocation_df = pd.DataFrame([
                {'Subject': subject, 'Weekly Blocks': blocks, 'Weekly Hours': blocks, 
                 'Difficulty': subjects[subject]}
                for subject, blocks in subject_allocation.items()
            ])
            
            col1, col2 = st.columns(2)
            with col1:
                fig_blocks = px.bar(
                    allocation_df,
                    x='Subject',
                    y='Weekly Blocks',
                    color='Difficulty',
                    color_continuous_scale='RdYlBu_r',
                    title='Weekly Study Blocks by Subject'
                )
                st.plotly_chart(fig_blocks, use_container_width=True)
            
            with col2:
                fig_pie = px.pie(
                    allocation_df,
                    values='Weekly Blocks',
                    names='Subject',
                    title='Study Time Distribution'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab4:
            st.subheader("💾 Export Your Schedule")
            
            # Create comprehensive export data
            export_blocks = []
            for day, blocks in timetable.items():
                for i, block in enumerate(blocks):
                    export_blocks.append({
                        'Day': day,
                        'Block_Number': i + 1,
                        'Subject': block['subject'],
                        'Study_Start': generator.format_time(int(block['study_start']), 
                                                           int((block['study_start'] % 1) * 60)),
                        'Study_End': generator.format_time(int(block['study_end']), 
                                                         int((block['study_end'] % 1) * 60)),
                        'Break_End': generator.format_time(int(block['break_end']), 
                                                         int((block['break_end'] % 1) * 60)),
                        'Difficulty': block['difficulty']
                    })
            
            if export_blocks:
                export_df = pd.DataFrame(export_blocks)
                csv_str = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Detailed Schedule (CSV)",
                    data=csv_str,
                    file_name=f"pomodoro_schedule_{datetime.date.today()}.csv",
                    mime="text/csv"
                )
            
            # Summary export
            summary_df = generator.create_daily_summary(timetable)
            summary_csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Daily Summary (CSV)",
                data=summary_csv,
                file_name=f"daily_summary_{datetime.date.today()}.csv",
                mime="text/csv"
            )
        
        # Study tips
        with st.expander("💡 Pomodoro Study Tips"):
            st.markdown("""
            **🎯 During Study Blocks (60 min):**
            - Remove all distractions (phone, social media)
            - Focus on ONE subject only
            - Take notes actively
            - If you finish early, review what you learned
            
            **☕ During Breaks (15 min):**
            - Step away from your study area
            - Hydrate and have a light snack
            - Do light stretching or walk around
            - Avoid screens to rest your eyes
            
            **📈 Maximizing Effectiveness:**
            - Start with your hardest subjects when energy is high
            - Stick to the timing religiously
            - Track which subjects need more/less time
            - Adjust difficulty ratings based on progress
            """)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🍅 Pomodoro Technique + Smart Scheduling + Google Calendar
        
        This tool combines the proven Pomodoro Technique with intelligent subject distribution and Google Calendar integration:
        
        ### ⏰ Pomodoro Structure:
        - **60 minutes** focused study per block
        - **15 minutes** break after each study session
        - **Ends by 9 PM** for healthy work-life balance
        - **Even distribution** of subjects throughout the week
        
        ### 📅 Google Calendar Integration:
        - **Automatic scheduling** of study blocks
        - **Smart reminders** 10 minutes and 1 hour before each session
        - **Detailed descriptions** with study tips and subject info
        - **Color-coded events** for easy identification
        
        ### 🧠 Smart Features:
        - **Difficulty-Based Priority**: Harder subjects get better time slots
        - **Automatic Scheduling**: Avoids your busy hours
        - **Balanced Distribution**: Prevents subject cramming
        - **Visual Timeline**: See your entire week at a glance
        
        ### 📊 Why It Works:
        - Maintains peak concentration for optimal learning
        - Prevents burnout with regular breaks
        - Builds consistent study habits
        - Scientifically proven to improve retention
        
        👈 **Start by entering your subjects and availability in the sidebar!**
        """)
        
        # Show example schedule
        st.info("""
        **Example Daily Schedule:**
        - 9:00 AM - 10:00 AM: Mathematics (Study)
        - 10:00 AM - 10:15 AM: Break
        - 10:15 AM - 11:15 AM: Physics (Study)  
        - 11:15 AM - 11:30 AM: Break
        - 2:00 PM - 3:00 PM: Chemistry (Study)
        - 3:00 PM - 3:15 PM: Break
        """)

if __name__ == "__main__":
    main()
