import io
import re
import string
import hashlib
import numpy as np
import streamlit as st

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
        vec = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=1, stop_words="english")
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
    st.title("📖 Study PDF: Summary, Full-Sentence Flashcards (with Robust Fallback), and Q&A")

    with st.sidebar:
        st.subheader("Options")
        global_summary_sents = st.slider("Summary sentences (whole PDF)", 6, 25, 12)
        cards_per_paragraph = st.slider("Flashcards per paragraph (primary)", 1, 25, 12)
        fallback_cards_total = st.slider("Fallback: total flashcards across PDF", 4, 40, 16)
        explain_min = st.slider("Flashcard explanation min words", 20, 60, 25)
        explain_max = st.slider("Flashcard explanation max words", 30, 120, 55)
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

    st.subheader("📌 Main Points Summary (Whole PDF)")
    summary_sents = summarize_main_points(all_sentences, sentence_count=global_summary_sents)
    if summary_sents:
        for s in summary_sents:
            st.markdown(f"- {s}")
    else:
        st.caption("No clear sentences found for a summary.")

    st.subheader("✨ Flashcards (Title is a complete sentence; click for a short context)")
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

    st.subheader("💬 Ask a Doubt (Answer comes only from this PDF)")
    needs_rebuild = st.session_state.get("qa_pdf_hash") != pdf_hash
    if needs_rebuild:
        vec, mat = build_qa_index(paragraphs if paragraphs else [cleaned])
        st.session_state.qa_index = (vec, mat)
        st.session_state.qa_paragraphs = paragraphs if paragraphs else [cleaned]
        st.session_state.qa_pdf_hash = pdf_hash

    user_q = st.text_input("Type your question about the PDF content...")
    if st.button("Get Answer", use_container_width=True):
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

    st.caption("This app tries per-paragraph flashcards first, then falls back to whole-PDF cards if needed. For scanned PDFs, run OCR to add a text layer.")

if __name__ == "__main__":
    main()
