import io
import re
import string
import hashlib
import numpy as np
import streamlit as st
import datetime
import json
from typing import Dict, List, Tuple
import math
import os
import pickle
import pytz
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import random
from collections import Counter

# -------------------- TIMETABLE GENERATOR --------------------

class PomodoroTimetableGenerator:
    def __init__(self, study_hours_per_day: int, break_duration: int = 5, long_break_duration: int = 15, sessions_before_long_break: int = 4):
        self.study_hours_per_day = study_hours_per_day
        self.break_duration = break_duration
        self.long_break_duration = long_break_duration
        self.sessions_before_long_break = sessions_before_long_break
        self.pomodoro_duration = 25  # minutes

    def generate_timetable(self, subjects: List[str], start_date: datetime.date, end_date: datetime.date, timezone: str = 'UTC') -> Dict[str, List[Dict[str, any]]]:
        tz = pytz.timezone(timezone)
        timetable = {}
        current_date = start_date

        while current_date <= end_date:
            daily_schedule = self._generate_daily_schedule(subjects, current_date, tz)
            timetable[current_date.isoformat()] = daily_schedule
            current_date += timedelta(days=1)

        return timetable

    def _generate_daily_schedule(self, subjects: List[str], date: datetime.date, tz: pytz.timezone) -> List[Dict[str, any]]:
        schedule = []
        total_study_minutes = self.study_hours_per_day * 60
        pomodoro_count = total_study_minutes // self.pomodoro_duration
        subject_sessions = self._distribute_sessions(subjects, pomodoro_count)

        start_time = datetime.datetime.combine(date, datetime.time(9, 0, tzinfo=tz))  # Start at 9 AM
        session_count = 0

        for subject, sessions in subject_sessions.items():
            for _ in range(sessions):
                session_count += 1
                end_time = start_time + timedelta(minutes=self.pomodoro_duration)

                schedule.append({
                    'subject': subject,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'type': 'study'
                })

                start_time = end_time

                # Add break
                if session_count % self.sessions_before_long_break == 0:
                    break_duration = self.long_break_duration
                else:
                    break_duration = self.break_duration

                end_time = start_time + timedelta(minutes=break_duration)
                schedule.append({
                    'subject': 'Break',
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'type': 'break'
                })
                start_time = end_time

        return schedule

    def _distribute_sessions(self, subjects: List[str], total_sessions: int) -> Dict[str, int]:
        base_sessions = total_sessions // len(subjects)
        remainder = total_sessions % len(subjects)

        distribution = {}
        for i, subject in enumerate(subjects):
            distribution[subject] = base_sessions + (1 if i < remainder else 0)

        return distribution

class GoogleCalendarIntegrator:
    def __init__(self, credentials_path: str = None):
        self.credentials_path = credentials_path or os.path.join(os.path.dirname(__file__), 'credentials.json')
        self.service = None

    def authenticate(self):
        """Authenticate with Google Calendar API"""
        creds = None
        token_path = os.path.join(os.path.dirname(__file__), 'token.pickle')

        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path,
                    ['https://www.googleapis.com/auth/calendar']
                )
                creds = flow.run_local_server(port=0)

            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)

        self.service = build('calendar', 'v3', credentials=creds)

    def create_events(self, timetable: Dict[str, List[Dict[str, any]]], calendar_id: str = 'primary'):
        """Create calendar events from timetable"""
        if not self.service:
            self.authenticate()

        created_events = []

        for date, events in timetable.items():
            for event in events:
                event_body = {
                    'summary': f"{event['type'].title()}: {event['subject']}",
                    'start': {
                        'dateTime': event['start_time'],
                        'timeZone': 'UTC',
                    },
                    'end': {
                        'dateTime': event['end_time'],
                        'timeZone': 'UTC',
                    },
                }

                if event['type'] == 'break':
                    event_body['colorId'] = '11'  # Red for breaks
                else:
                    event_body['colorId'] = '10'  # Green for study

                created_event = self.service.events().insert(
                    calendarId=calendar_id,
                    body=event_body
                ).execute()

                created_events.append(created_event)

        return created_events

# Optional: better PDF extractor (strongly recommended)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# Fallback extractor
try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

# Optional: TF-IDF (fallback implemented if not available)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# Optional: Speech Recognition
try:
    import speech_recognition as sr
    HAS_SR = True
except Exception:
    HAS_SR = False

# Optional: Google Calendar API
try:
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    HAS_GOOGLE_API = True
except Exception:
    HAS_GOOGLE_API = False


# -------------------- PDF TEXT EXTRACTION --------------------

def extract_text_from_pdf(filelike) -> str:
    # 1) PyMuPDF
    if HAS_PYMUPDF:
        try:
            with fitz.open(stream=filelike.read(), filetype="pdf") as doc:
                text_parts = []
                for page in doc:
                    txt = page.get_text("text") or ""
                    text_parts.append(txt)
                raw = "\n".join(text_parts)
                if len(raw.strip()) >= 200:
                    return raw
        except Exception:
            pass
        finally:
            try:
                filelike.seek(0)
            except Exception:
                pass

    # 2) PyPDF2
    if HAS_PYPDF2:
        try:
            reader = PyPDF2.PdfReader(filelike)
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    return ""
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            return "\n".join(text_parts)
        except Exception:
            return ""

    return ""


# -------------------- CLEANING / NORMALIZATION --------------------

UNWANTED_LINE_PATTERNS = re.compile(
    r"(?i)\b("
    r"publisher|author|edition|college|university|institute|department|faculty|"
    r"copyright|isbn|doi|website|index|contents|table of contents|"
    r"acknowledg(e)?ments|foreword|preface|biography|about the author|"
    r"figure\s*\d+|fig\.\s*\d+|table\s*\d+|page\s*\d+|pp\.\s*\d+|"
    r"references?|bibliography|appendix|supplement|abstract"
    r")\b"
)
UNWANTED_INLINE = ["\u00ad", "\uf0b7", "\u2022", "\u200b", "\u200c", "\u200d"]
BULLET_PREFIX_RE = re.compile(r"^\s*([\-–—•·●◦\*]|\d+[\.)])\s+")
ONLY_DECOR_RE = re.compile(r"^\s*[-–—=~_+*#]+\s*$")
REF_MARKER_RE = re.compile(r"\[\s*\d+\s*\]|\(\s*\d+\s*\)")
MULTI_SPACE_RE = re.compile(r"[ \t]+")

STOPWORDS = set("""
a an the and or but if while with without within into onto from to of in on at by for as that this these those there here
is are was were be been being have has had do does did can could should would may might will shall it its itself himself herself themselves
about above below under over between among per via etc such than then so not no nor also more most less least very much many few each either neither both
""".split())

def normalize_text(raw_text: str) -> str:
    for ch in UNWANTED_INLINE:
        raw_text = raw_text.replace(ch, " ")
    raw_text = re.sub(r"-\s*\n\s*(?=\w)", "", raw_text)  # de-hyphenate across linebreaks
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    raw_text = re.sub(r"[ \t]+\n", "\n", raw_text)
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
    return raw_text

def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if ONLY_DECOR_RE.match(line):
        return True
    if UNWANTED_LINE_PATTERNS.search(line):
        return True
    tokens = line.split()
    if len(tokens) <= 3:
        return True
    letters = sum(c.isalpha() for c in line)
    digits = sum(c.isdigit() for c in line)
    if letters == 0:
        return True
    alpha_ratio = letters / max(1, len(line))
    digit_ratio = digits / max(1, len(line))
    if alpha_ratio < 0.55 or digit_ratio > 0.25:
        return True
    if len(line) >= 12 and line.upper() == line and any(c.isalpha() for c in line):
        return True
    return False

def remove_repeating_headers_footers(lines):
    freq = {}
    for ln in lines:
        key = ln.lower()
        freq[key] = freq.get(key, 0) + 1
    max_allowed = max(2, len(lines) // 40)
    return [ln for ln in lines if freq.get(ln.lower(), 0) <= max_allowed]

def clean_text(raw_text: str) -> str:
    text = normalize_text(raw_text)
    if not text.strip():
        return ""
    lines = [MULTI_SPACE_RE.sub(" ", ln.strip()) for ln in text.split("\n")]

    cleaned = []
    for ln in lines:
        if not ln:
            continue
        ln = BULLET_PREFIX_RE.sub("", ln)
        ln = REF_MARKER_RE.sub("", ln)
        ln = re.sub(r"\b\S+@\S+\b", "", ln)
        ln = re.sub(r"\bhttps?://\S+\b", "", ln)
        ln = re.sub(r"\bwww\.\S+\b", "", ln)
        ln = ln.strip(string.punctuation + " ")
        ln = MULTI_SPACE_RE.sub(" ", ln).strip()
        if ln:
            cleaned.append(ln)

    cleaned = remove_repeating_headers_footers(cleaned)
    cleaned = [ln for ln in cleaned if not is_noise_line(ln)]

    paragraphs = []
    current = []
    for ln in cleaned:
        if not ln:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(ln)
    if current:
        paragraphs.append(" ".join(current))

    final_text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    final_text = MULTI_SPACE_RE.sub(" ", final_text).strip()
    return final_text


# -------------------- CHUNKING / SENTENCES --------------------

SENT_SPLIT = re.compile(r"(?<=[.?!])\s+(?=[A-Z(])")

def split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 20]

def split_into_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    out = []
    for s in sents:
        if REF_MARKER_RE.search(s):
            s = REF_MARKER_RE.sub("", s).strip()
        if len(s.split()) < 6:
            continue
        letters = sum(c.isalpha() for c in s)
        digits = sum(c.isdigit() for c in s)
        if letters == 0:
            continue
        alpha_ratio = letters / max(1, len(s))
        digit_ratio = digits / max(1, len(s))
        if alpha_ratio < 0.55 or digit_ratio > 0.25:
            continue
        if len(s) >= 12 and s.upper() == s and any(c.isalpha() for c in s):
            continue
        if not s.endswith((".", "?", "!")):
            s = s + "."
        out.append(s)
    return out


# -------------------- VECTORS --------------------

def build_vectorizer(texts: list[str]):
    if HAS_SKLEARN:
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
        mat = vec.fit_transform(texts)
        return vec, mat
    vocab = {}
    rows = []
    for txt in texts:
        tokens = [w for w in re.findall(r"[A-Za-z]{2,}", txt.lower()) if w not in STOPWORDS]
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
            if t not in vocab:
                vocab[t] = len(vocab)
        rows.append(counts)
    mat = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, counts in enumerate(rows):
        for t, c in counts.items():
            j = vocab[t]
            mat[i, j] = c
        if mat[i].sum() > 0:
            mat[i] /= (np.linalg.norm(mat[i]) + 1e-9)
    return (vocab, None), mat

def vectorize(vec, texts: list[str]):
    if HAS_SKLEARN and isinstance(vec, TfidfVectorizer):
        return vec.transform(texts)
    vocab, _ = vec
    mat = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, txt in enumerate(texts):
        tokens = [w for w in re.findall(r"[A-Za-z]{2,}", txt.lower()) if w not in STOPWORDS]
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            if t in vocab:
                j = vocab[t]
                mat[i, j] = c
        if mat[i].sum() > 0:
            mat[i] /= (np.linalg.norm(mat[i]) + 1e-9)
    return mat

def cos_sim(A, B):
    if HAS_SKLEARN and "csr_matrix" in str(type(A)):
        return cosine_similarity(A, B)
    if A.ndim == 1: A = A.reshape(1, -1)
    if B.ndim == 1: B = B.reshape(1, -1)
    denom = (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9) * (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A @ B.T) / denom


# -------------------- SUMMARY (GLOBAL MAIN POINTS) --------------------

def summarize_main_points(all_sentences: list[str], sentence_count: int = 10) -> list[str]:
    if not all_sentences:
        return []
    vec, mat = build_vectorizer(all_sentences)
    if HAS_SKLEARN and "csr_matrix" in str(type(mat)):
        centroid = mat.mean(axis=0)
        scores = (mat @ centroid.T).A.ravel()
    else:
        centroid = mat.mean(axis=0, keepdims=True)
        scores = (mat @ centroid.T).ravel()
    idxs_sorted = np.argsort(-scores).tolist()
    selected = []
    selected_vecs = []
    for idx in idxs_sorted:
        if len(selected) >= sentence_count:
            break
        cand_vec = vectorize(vec, [all_sentences[idx]])
        redundant = False
        for sv in selected_vecs:
            if cos_sim(cand_vec, sv)[0,0] > 0.75:
                redundant = True
                break
        if not redundant:
            selected.append(all_sentences[idx])
            selected_vecs.append(cand_vec)
    return selected


# -------------------- FLASHCARDS --------------------

def sentence_is_similar_to_any(target: str, others: list[str], threshold: float = 0.8) -> bool:
    if not others:
        return False
    vec, mat = build_vectorizer(others + [target])
    q = vectorize(vec, [target])
    pool = vectorize(vec, others)
    if pool.shape[0] == 0:
        return False
    sims = cos_sim(pool, q).ravel()
    return float(np.max(sims)) >= threshold

def make_short_explanation_from_context(paragraph: str, anchor_sentence: str,
                                        min_words: int = 25, max_words: int = 55) -> str:
    sents = split_into_sentences(paragraph)
    if not sents:
        return anchor_sentence
    try:
        idx = sents.index(anchor_sentence)
    except ValueError:
        idx = 0
    window = sents[max(0, idx-1): min(len(sents), idx+2)]
    text = " ".join(window).strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;: ") + "."
    elif len(words) < min_words and idx+2 < len(sents):
        text = (text + " " + sents[min(len(sents)-1, idx+2)]).strip()
    if not text.endswith((".", "?", "!")):
        text += "."
    return text

def generate_paragraph_flashcards(paragraph: str, banned_sentences: list[str],
                                  limit: int, explain_min: int, explain_max: int) -> list[dict]:
    sentences = split_into_sentences(paragraph)
    if not sentences:
        return []
    vec, mat = build_vectorizer(sentences)
    if HAS_SKLEARN and "csr_matrix" in str(type(mat)):
        centroid = mat.mean(axis=0)
        scores = (mat @ centroid.T).A.ravel()
    else:
        centroid = mat.mean(axis=0, keepdims=True)
        scores = (mat @ centroid.T).ravel()
    order = np.argsort(-scores).tolist()

    selected = []
    selected_vecs = []
    for idx in order:
        if len(selected) >= limit:
            break
        s = sentences[idx]
        if banned_sentences and sentence_is_similar_to_any(s, banned_sentences, threshold=0.8):
            continue
        cand_vec = vectorize(vec, [s])
        redundant = False
        for sv in selected_vecs:
            if cos_sim(cand_vec, sv)[0, 0] > 0.7:
                redundant = True
                break
        if redundant:
            continue
        selected.append(s)
        selected_vecs.append(cand_vec)

    cards = []
    for s in selected:
        title = s  # full sentence
        explanation = make_short_explanation_from_context(paragraph, s, explain_min, explain_max)
        cards.append({"summary": title, "explanation": explanation})
    return cards

def generate_global_fallback_flashcards(all_text: str, banned_sentences: list[str],
                                        desired_count: int, explain_min: int, explain_max: int) -> list[dict]:
    # Fallback: sample flashcards from the whole document when paragraph-level fails
    sentences = split_into_sentences(all_text)
    if not sentences:
        return []
    vec, mat = build_vectorizer(sentences)
    if HAS_SKLEARN and "csr_matrix" in str(type(mat)):
        centroid = mat.mean(axis=0)
        scores = (mat @ centroid.T).A.ravel()
    else:
        centroid = mat.mean(axis=0, keepdims=True)
        scores = (mat @ centroid.T).ravel()
    order = np.argsort(-scores).tolist()

    selected = []
    selected_vecs = []
    for idx in order:
        if len(selected) >= desired_count:
            break
        s = sentences[idx]
        if banned_sentences and sentence_is_similar_to_any(s, banned_sentences, threshold=0.8):
            continue
        cand_vec = vectorize(vec, [s])
        redundant = False
        for sv in selected_vecs:
            if cos_sim(cand_vec, sv)[0, 0] > 0.7:
                redundant = True
                break
        if redundant:
            continue
        selected.append(s)
        selected_vecs.append(cand_vec)

    cards = []
    for s in selected:
        title = s
        # For fallback, explanation = the same sentence (already concise)
        explanation = s if len(s.split()) >= explain_min else (s + " " + s)
        words = explanation.split()
        if len(words) > explain_max:
            explanation = " ".join(words[:explain_max]).rstrip(",;: ") + "."
        if not explanation.endswith((".", "?", "!")):
            explanation += "."
        cards.append({"summary": title, "explanation": explanation})
    return cards


# -------------------- QUIZ HELPER FUNCTIONS (PORTED FROM JS) --------------------

import random
from collections import Counter
import re

STOPWORDS_QUIZ = set("""
the a an and or but if then else for to of in on by with as at from that this these those it its is are was were be been being which who whom whose what when where why how not so into we you they he she i their our your his her them us me
can could should would may might will shall do does did done have has had more most many much very such other than also because however therefore thus hence consequently in contrast for example whereas while
""".split())

def word_count(text: str) -> int:
    return len(text.split())

def clamp(n: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(max_val, n))

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]{2,}", text.lower())
    return [w for w in tokens if w not in STOPWORDS_QUIZ]

def extract_top_terms_with_phrases(text: str, max_terms: int = 400) -> list[dict]:
    tokens = tokenize(text)
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    is_good = lambda t: t not in STOPWORDS_QUIZ and len(t) >= 3 and re.match(r"[a-z]", t)
    for i in range(len(tokens)):
        t1 = tokens[i]
        if is_good(t1):
            unigrams[t1] += 1
        if i + 1 < len(tokens):
            t2 = tokens[i + 1]
            if is_good(t1) and is_good(t2):
                bigrams[f"{t1} {t2}"] += 1
        if i + 2 < len(tokens):
            t2 = tokens[i + 1]
            t3 = tokens[i + 2]
            if is_good(t1) and is_good(t2) and is_good(t3):
                trigrams[f"{t1} {t2} {t3}"] += 1
    scored = []
    for term, count in unigrams.items():
        scored.append({"term": term, "count": count, "len": 1, "score": count})
    for term, count in bigrams.items():
        scored.append({"term": term, "count": count, "len": 2, "score": count * 1.6})
    for term, count in trigrams.items():
        scored.append({"term": term, "count": count, "len": 3, "score": count * 2.2})
    scored.sort(key=lambda x: x["score"], reverse=True)
    taken = set()
    result = []
    for t in scored:
        key = t["term"]
        if any(key in existing or existing in key for existing in taken):
            continue
        taken.add(key)
        result.append(t)
        if len(result) >= max_terms:
            break
    return result

def pick_informative_sentences(sentences: list[str]) -> list[dict]:
    scored = []
    for idx, s in enumerate(sentences):
        wc = word_count(s)
        has_comma = "," in s
        has_link = re.search(r"\b(therefore|however|because|whereas|although|while|thus|hence|consequently|in contrast|for example)\b", s, re.I)
        score = 0
        if 12 <= wc <= 35:
            score += 2
        elif 8 <= wc <= 45:
            score += 1
        if has_comma:
            score += 1
        if has_link:
            score += 1.2
        if re.search(r"[;:]", s):
            score += 0.5
        scored.append({"s": s, "idx": idx, "wc": wc, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def jaccard_characters(a: str, b: str) -> float:
    sa = set(a)
    sb = set(b)
    inter = sa.intersection(sb)
    union = sa.union(sb)
    return len(inter) / len(union) if union else 0

def title_case(s: str) -> str:
    return s.title()

def escape_regexp(s: str) -> str:
    return re.escape(s)

def choose_targets(sentences: list[str], top_terms: list[dict], want: int, rng: random.Random) -> list[dict]:
    informative = pick_informative_sentences(sentences)
    term_set = set(t["term"] for t in top_terms)
    selected = []
    used_sent_idx = set()
    used_terms = set()
    for item in informative:
        if len(selected) >= want * 2:
            break
        if item["idx"] in used_sent_idx:
            continue
        s = item["s"]
        tokens = tokenize(s)
        candidates = []
        for length in range(3, 0, -1):
            for i in range(len(tokens) - length + 1):
                phrase = " ".join(tokens[i:i + length])
                if phrase in term_set and len(phrase) >= 4 and not re.match(r"^\d+$", phrase):
                    candidates.append({"phrase": phrase, "len": length})
            if candidates:
                break
        if not candidates:
            continue
        chosen = None
        for cand in candidates:
            if cand["phrase"] not in used_terms:
                chosen = cand
                break
        if not chosen:
            continue
        selected.append({"idx": item["idx"], "sentence": s, "answer": chosen["phrase"]})
        used_sent_idx.add(item["idx"])
        used_terms.add(chosen["phrase"])
    return selected[:want]

def nearest_sentences(sentences: list[str], idx: int, radius: int = 2) -> list[dict]:
    res = []
    for i in range(max(0, idx - radius), min(len(sentences), idx + radius + 1)):
        if i != idx:
            res.append({"s": sentences[i], "idx": i})
    return res

def build_distractors(answer: str, pool: list[str], sentences: list[str], anchor_idx: int, k: int, rng: random.Random) -> list[str]:
    answer_lc = answer.lower()
    answer_len = len(answer_lc)
    nearby_text = " ".join([ns["s"] for ns in nearest_sentences(sentences, anchor_idx, 2)])
    nearby_terms = [t["term"] for t in extract_top_terms_with_phrases(nearby_text, 60)]
    all_candidates = list(set(nearby_terms + pool))
    candidates = [
        t for t in all_candidates
        if t.lower() != answer_lc
        and abs(len(t) - answer_len) <= 5
        and jaccard_characters(t.lower(), answer_lc) < 0.65
        and t.lower() not in STOPWORDS_QUIZ
        and not re.match(r"^\d+$", t)
    ]
    rng.shuffle(candidates)
    unique = []
    for c in candidates:
        if len(unique) >= k:
            break
        if any(jaccard_characters(u.lower(), c.lower()) >= 0.65 for u in unique):
            continue
        unique.append(c)
    return [title_case(u) for u in unique]

def shuffle_and_label(options: list[str], rng: random.Random) -> list[dict]:
    copy = options[:]
    rng.shuffle(copy)
    return [{"label": chr(65 + i), "text": opt} for i, opt in enumerate(copy)]

def make_cloze_question(target: dict, sentences: list[str], top_pool: list[str], rng: random.Random) -> dict:
    answer = target["answer"]
    rx = re.compile(r"\b" + re.escape(answer) + r"\b", re.I)
    if not rx.search(target["sentence"]):
        return None
    prompt = rx.sub("_____", target["sentence"])
    if word_count(prompt) < 10:
        return None
    distractors = build_distractors(answer, top_pool, sentences, target["idx"], 3, rng)
    if len(distractors) < 3:
        return None
    options_raw = shuffle_and_label([title_case(answer)] + distractors, rng)
    correct_index = next(i for i, o in enumerate(options_raw) if o["text"].lower() == title_case(answer).lower())
    options = [f"{o['label']}) {o['text']}" for o in options_raw]
    explanation = f"Context: {build_context_snippet(sentences, target['idx'], answer)}"
    return {
        "type": "cloze",
        "prompt": prompt,
        "options": options,
        "correctIndex": correct_index,
        "explanation": explanation
    }

def definition_pattern(sentence: str, answer: str) -> str:
    rx = re.compile(rf"\b{re.escape(answer)}\b\s+(is|are|refers to|means|denotes|involves|consists of|can be defined as)\s+([^.;:]+)", re.I)
    m = rx.search(sentence)
    if m:
        return f"{title_case(answer)} {m.group(1)} {m.group(2).strip()}."
    return None

def make_definition_question(target: dict, sentences: list[str], top_pool: list[str], rng: random.Random) -> dict:
    answer = title_case(target["answer"])
    cand_sentences = [target["sentence"]] + [ns["s"] for ns in nearest_sentences(sentences, target["idx"], 1)]
    def_text = None
    for s in cand_sentences:
        def_text = definition_pattern(s, target["answer"])
        if def_text:
            break
    if not def_text:
        return None
    distractors = build_distractors(target["answer"], top_pool, sentences, target["idx"], 3, rng)
    distractors = [f"{title_case(dt)} is {generate_generic_definition_tail(dt)}" for dt in distractors]
    if len(distractors) < 3:
        return None
    options_raw = shuffle_and_label([def_text] + distractors, rng)
    correct_index = next(i for i, o in enumerate(options_raw) if o["text"] == def_text)
    options = [f"{o['label']}) {o['text']}" for o in options_raw]
    explanation = f"Context: {build_context_snippet(sentences, target['idx'], target['answer'])}"
    return {
        "type": "definition",
        "prompt": f"Which option best defines {answer}?",
        "options": options,
        "correctIndex": correct_index,
        "explanation": explanation
    }

def make_true_about_question(target: dict, sentences: list[str], top_pool: list[str], rng: random.Random) -> dict:
    answer = title_case(target["answer"])
    s = target["sentence"]
    if word_count(s) < 10:
        return None
    rx_replace = re.compile(r"\b" + re.escape(target["answer"]) + r"\b", re.I)
    true_stmt = s if rx_replace.search(s) else rx_replace.sub(answer, s)
    false_stmts = []
    distractor_terms = build_distractors(target["answer"], top_pool, sentences, target["idx"], 6, rng)
    for dt in distractor_terms[:6]:
        wrong = re.sub(r"\b" + re.escape(answer) + r"\b", dt, true_stmt, flags=re.I)
        if wrong != true_stmt and word_count(wrong) >= 10:
            false_stmts.append(wrong)
        if len(false_stmts) >= 3:
            break
    if len(false_stmts) < 3:
        return None
    options_raw = shuffle_and_label([true_stmt] + false_stmts, rng)
    correct_index = next(i for i, o in enumerate(options_raw) if o["text"] == true_stmt)
    options = [f"{o['label']}) {o['text']}" for o in options_raw]
    explanation = f"Context: {build_context_snippet(sentences, target['idx'], target['answer'])}"
    return {
        "type": "trueabout",
        "prompt": f"Which of the following is true about {answer}?",
        "options": options,
        "correctIndex": correct_index,
        "explanation": explanation
    }

def generate_generic_definition_tail(term: str) -> str:
    tails = [
        "primarily concerned with peripheral aspects unrelated to core concepts",
        "a general approach focusing on adjacent but distinct principles",
        "characterized by features that are context-dependent rather than essential",
        "commonly associated with outcomes rather than underlying mechanisms"
    ]
    return random.choice(tails)

def build_context_snippet(sentences: list[str], idx: int, term: str, max_len: int = 240) -> str:
    parts = []
    if idx > 0:
        parts.append(sentences[idx - 1])
    parts.append(sentences[idx])
    if idx < len(sentences) - 1:
        parts.append(sentences[idx + 1])
    full = re.sub(r"\s+", " ", " ".join(parts)).strip()
    snippet = full
    if len(snippet) > max_len:
        rx = re.compile(r"\b" + re.escape(term) + r"\b", re.I)
        m = rx.search(snippet)
        if m and m.start() is not None:
            center = m.start() + len(m.group(0)) / 2
            start = max(0, int(center - max_len / 2))
            snippet = ("" if start == 0 else "… ") + snippet[start:start + max_len].strip() + ("" if start + max_len >= len(full) else " …")
        else:
            snippet = snippet[:max_len].strip() + " …"
    return highlight_term(snippet, term)

def highlight_term(text: str, term: str) -> str:
    rx = re.compile(r"(\b" + re.escape(term) + r"\b)", re.I)
    return rx.sub(r'"\1"', text)

def generate_moderate_questions(text: str, num_questions: int, shuffle_seed: int = None) -> list[dict]:
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
    else:
        rng = random.Random()
    sentences = split_into_sentences(text)
    top_terms = extract_top_terms_with_phrases(text, 400)
    top_pool = [t["term"] for t in top_terms][:250]
    raw_targets = choose_targets(sentences, top_terms, num_questions * 3, rng)
    if not raw_targets:
        return []
    questions = []
    used_prompts = set()
    want_cloze = max(1, num_questions // 3)
    want_def = max(1, num_questions // 3)
    want_true = max(1, num_questions // 3)
    rng.shuffle(raw_targets)

    def try_add(q: dict) -> bool:
        if not q or q["prompt"] in used_prompts or not q.get("options") or len(q["options"]) != 4:
            return False
        used_prompts.add(q["prompt"])
        questions.append(q)
        return True

    # Prioritize types
    for t in raw_targets:
        if len([q for q in questions if q["type"] == "cloze"]) >= want_cloze:
            break
        try_add(make_cloze_question(t, sentences, top_pool, rng))
    for t in raw_targets:
        if len([q for q in questions if q["type"] == "definition"]) >= want_def:
            break
        try_add(make_definition_question(t, sentences, top_pool, rng))
    for t in raw_targets:
        if len([q for q in questions if q["type"] == "trueabout"]) >= want_true:
            break
        try_add(make_true_about_question(t, sentences, top_pool, rng))

    # Fill remaining
    for t in raw_targets:
        if len(questions) >= num_questions:
            break
        attempts = [
            make_cloze_question(t, sentences, top_pool, rng),
            make_true_about_question(t, sentences, top_pool, rng),
            make_definition_question(t, sentences, top_pool, rng)
        ]
        for q in attempts:
            if len(questions) >= num_questions:
                break
            try_add(q)
    return questions[:num_questions]


# -------------------- QA (RETRIEVAL, NO LLM) --------------------

def build_qa_index(paragraphs: list[str]):
    if not paragraphs:
        return None, None
    vec, mat = build_vectorizer(paragraphs)
    return vec, mat

def answer_question(query: str, paragraphs: list[str], vec, mat, top_k: int = 3) -> dict:
    if not query.strip() or not paragraphs:
        return {"answer": "No content available.", "sources": []}
    qv = vectorize(vec, [query])
    sims = cos_sim(mat, qv).ravel()
    top_idx = np.argsort(-sims)[:top_k].tolist()
    chosen = [paragraphs[i] for i in top_idx if sims[i] > 0.05]

    sent_pool = []
    for p in chosen:
        sent_pool.extend(split_into_sentences(p))
    if not sent_pool:
        return {"answer": "I could not find a relevant answer in the PDF.", "sources": []}

    svec, smat = build_vectorizer(sent_pool)
    qsv = vectorize(svec, [query])
    ssims = cos_sim(smat, qsv).ravel()
    best_sent_idx = np.argsort(-ssims)[:5]
    best_sents = [sent_pool[i] for i in best_sent_idx if ssims[i] > 0.05]

    answer = " ".join(best_sents) if best_sents else "I could not find a clear answer in the PDF."
    return {"answer": answer, "sources": chosen}


# -------------------- STREAMLIT APP --------------------

def main():
    st.title("📖 Study PDF: Summary, Full-Sentence Flashcards, Quiz, and Q&A")

    # Initialize session state for quiz
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = None
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = None

    with st.sidebar:
        st.subheader("Options")
        global_summary_sents = st.slider("Summary sentences (whole PDF)", 6, 25, 12)
        cards_per_paragraph = st.slider("Flashcards per paragraph (primary)", 1, 25, 12)
        fallback_cards_total = st.slider("Fallback: total flashcards across PDF", 4, 40, 16)
        explain_min = st.slider("Flashcard explanation min words", 20, 60, 25)
        explain_max = st.slider("Flashcard explanation max words", 30, 120, 55)
        num_questions = st.slider("Quiz questions", 3, 50, 10)
        relax_filters = st.checkbox("Fallback sooner if few cards found", value=True)
        st.caption("If no/low flashcards per paragraph, a global fallback will generate cards from the whole PDF.")

    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if not uploaded:
        st.info("Upload a PDF to begin.")
        return

    pdf_bytes = uploaded.getvalue()
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

    raw = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    if not raw or len(raw.strip()) < 100:
        st.error("Could not read text from this PDF. If it is scanned, OCR it first (e.g., ocrmypdf).")
        return

    cleaned = clean_text(raw)
    if not cleaned or len(cleaned.split()) < 60:
        st.error("Could not extract meaningful content after cleaning. Try another (text-based) PDF or OCR it.")
        return

    paragraphs = split_into_paragraphs(cleaned)
    all_sentences = split_into_sentences(cleaned)

    tab_summary, tab_flashcards, tab_quiz, tab_qa, tab_timetable = st.tabs(["📌 Summary", "✨ Flashcards", "🧠 Quiz", "💬 Q&A", "📅 Timetable"])

    with tab_summary:
        st.subheader("Main Points Summary (Whole PDF)")
        summary_sents = summarize_main_points(all_sentences, sentence_count=global_summary_sents)
        if summary_sents:
            for s in summary_sents:
                st.markdown(f"- {s}")
        else:
            st.caption("No clear sentences found for a summary.")

    with tab_flashcards:
        st.subheader("Flashcards (Title is a complete sentence; click for a short context)")
        summary_sents = summarize_main_points(all_sentences, sentence_count=global_summary_sents)  # Reuse for banned
        generated_any = False
        total_cards_count = 0

        # Primary: per paragraph
        for p_idx, para in enumerate(paragraphs, start=1):
            cards = generate_paragraph_flashcards(
                para,
                banned_sentences=summary_sents,
                limit=cards_per_paragraph,
                explain_min=explain_min,
                explain_max=explain_max
            )
            if not cards:
                continue
            generated_any = True
            total_cards_count += len(cards)
            st.markdown(f"#### Paragraph {p_idx}")
            for i, card in enumerate(cards, start=1):
                with st.expander(f"Flashcard {i}: {card['summary']}", expanded=False):
                    st.markdown(card["explanation"])
            st.markdown("---")

        # Fallback: if none or too few
        need_fallback = (not generated_any) or (relax_filters and total_cards_count < max(6, fallback_cards_total // 2))
        if need_fallback:
            fallback_needed_count = max(0, fallback_cards_total - total_cards_count)
            if fallback_needed_count > 0:
                st.info("Generating fallback flashcards from the whole PDF...")
                cards = generate_global_fallback_flashcards(
                    cleaned,
                    banned_sentences=summary_sents,
                    desired_count=fallback_needed_count,
                    explain_min=explain_min,
                    explain_max=explain_max
                )
                if cards:
                    generated_any = True
                    st.markdown("#### Fallback Flashcards")
                    for i, card in enumerate(cards, start=1):
                        with st.expander(f"Flashcard F{i}: {card['summary']}", expanded=False):
                            st.markdown(card["explanation"])
                    st.markdown("---")

        if not generated_any:
            st.warning("Still no flashcards. This PDF may be mostly images or heavily formatted. Try an OCRed/text-based PDF.")

    with tab_quiz:
        st.subheader("Auto Quiz from PDF (Moderate Difficulty)")
        if st.button("Generate Quiz"):
            questions = generate_moderate_questions(cleaned, num_questions)
            if questions:
                st.session_state.quiz_questions = questions
                st.session_state.quiz_answers = {}
                st.session_state.quiz_score = None
                st.success(f"Generated {len(questions)} questions.")
            else:
                st.warning("Could not generate questions. Try a different PDF or reduce the number of questions.")
                st.session_state.quiz_questions = None

        if st.session_state.quiz_questions:
            questions = st.session_state.quiz_questions
            answers = st.session_state.quiz_answers
            score = st.session_state.quiz_score

            for i, q in enumerate(questions):
                st.markdown(f"**Question {i+1}:** {q['prompt']}")
                selected_option = st.radio(
                    "Choose the correct option:",
                    q['options'],
                    index=answers.get(i, None),
                    key=f"quiz_radio_{i}",
                    format_func=lambda x: x
                )
                if selected_option is not None:
                    answers[i] = q['options'].index(selected_option)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("Submit Quiz"):
                    correct = 0
                    total = len(questions)
                    feedback = []
                    for i, q in enumerate(questions):
                        user_ans = answers.get(i, -1)
                        correct_ans = q['correctIndex']
                        if user_ans == correct_ans:
                            correct += 1
                            feedback.append(f"Question {i+1}: Correct")
                        else:
                            feedback.append(f"Question {i+1}: Incorrect. Correct answer: {q['options'][correct_ans]}")
                            if q['explanation']:
                                feedback[-1] += f"\nExplanation: {q['explanation']}"
                    st.session_state.quiz_score = (correct, total)
                    st.session_state.feedback = feedback
            with col2:
                if st.button("Reset Quiz"):
                    st.session_state.quiz_questions = None
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_score = None
                    st.rerun()

            if score is not None:
                correct, total = score
                st.metric("Score", f"{correct}/{total}")
                with st.expander("Detailed Feedback"):
                    for fb in st.session_state.feedback:
                        st.write(fb)

    with tab_qa:
        st.subheader("Ask a Doubt (Answer comes only from this PDF)")
        needs_rebuild = st.session_state.get("qa_pdf_hash") != pdf_hash
        if needs_rebuild:
            vec, mat = build_qa_index(paragraphs if paragraphs else [cleaned])
            st.session_state.qa_index = (vec, mat)
            st.session_state.qa_paragraphs = paragraphs if paragraphs else [cleaned]
            st.session_state.qa_pdf_hash = pdf_hash

        def update_q():
            st.session_state['user_q'] = st.session_state['user_q_input']

        user_q = st.session_state.get('user_q', '')
        col1, col2 = st.columns([4, 1])
        with col1:
            user_q_input = st.text_input("Type your question about the PDF content...", value=user_q, key='user_q_input', on_change=update_q)
        with col2:
            if HAS_SR:
                audio = st.audio_input("🎤", key='audio_input')
                if audio is not None:
                    r = sr.Recognizer()
                    try:
                        audio_data = sr.AudioData(audio, sample_rate=16000, sample_width=2)
                        text = r.recognize_google(audio_data)
                        st.session_state['user_q'] = text
                        st.success(f"Transcribed: {text}")
                        st.rerun()
                    except sr.UnknownValueError:
                        st.error("Could not understand the audio.")
                    except sr.RequestError as e:
                        st.error(f"Could not request results; {e}")
            else:
                st.caption("Voice recognition not available.")

        if st.button("Get Answer", use_container_width=True):
            user_q = st.session_state.get('user_q', '').strip()
            if not user_q:
                st.warning("Please enter or speak a question.")
                return
            vec, mat = st.session_state.get("qa_index", (None, None))
            if vec is None or mat is None:
                st.error("QA index not available for this PDF.")
            else:
                result = answer_question(user_q, st.session_state.qa_paragraphs, vec, mat, top_k=3)
                st.success(result["answer"])
                if result["sources"]:
                    with st.expander("Show supporting excerpts"):
                        for j, src in enumerate(result["sources"], start=1):
                            st.markdown(f"**Source {j}:** {src}")

    st.caption("This app tries per-paragraph flashcards first, then falls back to whole-PDF cards if needed. For scanned PDFs, run OCR to add a text layer. Quiz generates moderate MCQs from key concepts.")

if __name__ == "__main__":
    main()
