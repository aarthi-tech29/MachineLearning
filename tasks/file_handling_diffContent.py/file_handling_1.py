# ======================================================================
import pdfplumber
from docx import Document
import pandas as pd
import re

# =========================
# SYNONYMS
# =========================
SYNONYMS = {
    # =========================
    # ML / DL / AI
    # =========================
    "ml": "machine learning",
    "machinelearning": "machine learning",
    "learning model": "machine learning",
    "model training": "machine learning",
    "training model": "machine learning",

    "dl": "deep learning",
    "deeplearning": "deep learning",

    "ai": "artificial intelligence",
    "artificialintelligence": "artificial intelligence",

    # =========================
    # DB / DBMS
    # =========================
    "db": "database",
    "database": "database",
    "dbms": "database management system",
    "database management system": "database management system",
    "database system": "database management system",
    "sql": "structured query language",
    "structured query language": "structured query language",

    # =========================
    # PYTHON
    # =========================
    "python": "python",
    "py": "python",
    "programming language": "python",
    "coding language": "python",
    "script": "python",

    # =========================
    # PROGRAMMING CONCEPTS
    # =========================
    "function": "function",
    "functions": "function",
    "method": "function",
    "variable": "variable",
    "loop": "loop",
    "iteration": "loop",

    "list": "list",
    "tuple": "tuple",
    "dictionary": "dictionary",
    "dict": "dictionary",

    # =========================
    # USAGE WORDS
    # =========================
    "use": "used",
    "uses": "used",
    "used for": "used",
    "purpose": "used",
    "application": "used",
    "benefit": "used",
    "consumed": "used",
    "consume": "used",

    # =========================
    # SAFE QUESTION NORMALIZATION
    # =========================
    "explain": "what",
    "describe": "what",
    "tell": "what",
    "define": "what",

    # =========================
    # EXTRA SAFE EXTENSIONS YOU ASKED
    # =========================
    "working": "used",
    "working of": "used",
    "role": "used",
    "how does": "used",
    "how do": "used",
    "meaning": "what",
    "about": "what",
    "purpose of": "used",
    "application of": "used"
}
# =========================
# CLEAN / NORMALIZE
# =========================
def clean(text):
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def normalize(text):
    text = clean(text)
    for k, v in SYNONYMS.items():
        text = re.sub(r'\b' + re.escape(k) + r'\b', v, text)
    return text

# =========================
# CHECKS
# =========================
def is_question(text):
    return text.lower().startswith(("what", "how", "why", "when", "where", "who"))

def is_single_word(text):
    return len(clean(text).split()) == 1

# IMPORTANT FIX: STRICT BLOCK FOR LEARNING
def is_learning_query(text):
    q = clean(text)
    return q in {
        "what is learning",
        "learning",
        "define learning",
        "explain learning",
        "what learning"
    }

# =========================
# LOAD FILE DATA
# =========================
def extract_qa(lines):
    qa = []
    q, a = None, []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_question(line):
            if q and a:
                qa.append((normalize(q), " ".join(a)))
            q = line
            a = []
        else:
            if q:
                a.append(line)

    if q and a:
        qa.append((normalize(q), " ".join(a)))

    return qa


def read_pdf(file):
    lines = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                lines.extend(t.split("\n"))
    return extract_qa(lines)


def read_docx(file):
    doc = Document(file)
    return extract_qa([p.text for p in doc.paragraphs])


def read_csv(file):
    df = pd.read_csv(file)
    return [(normalize(str(r.iloc[0])), str(r.iloc[1])) for _, r in df.iterrows()]


# =========================
# LOAD DATA
# =========================
qa_data = []
qa_data.extend(read_pdf("data.pdf"))
qa_data.extend(read_docx("data.docx"))
qa_data.extend(read_csv("data.csv"))

print(f"Loaded {len(qa_data)} Q&A pairs.")

# =========================
# ANSWER ENGINE
# =========================
def answer_question(user_question):

    q_raw = clean(user_question)

    # SINGLE WORD
    if is_single_word(user_question):
        return "Please be more specific."

    #  FIX: BLOCK LEARNING BEFORE SEARCH
    if is_learning_query(user_question):
        return "Please be more specific."

    #  RANDOM BLOCK
    random_block = ["cricket", "music", "dhoni", "dance", "invented"]
    if any(x in q_raw for x in random_block):
        return "Answer not found the questions not available in files"

    q = normalize(user_question)
    q_words = set(q.split())

    best_answer = None
    best_score = 0

    for stored_q, answer in qa_data:

        stored_words = set(stored_q.split())
        common = q_words.intersection(stored_words)

        if len(common) == 0:
            continue

        score = len(common)

        # exact match boost
        if q == stored_q:
            score += 10

        # phrase match boost
        if q in stored_q or stored_q in q:
            score += 5

        # domain boost
        if "python" in stored_q and "python" in q:
            score += 3

        if "used" in stored_q and "used" in q:
            score += 2

        if score > best_score:
            best_score = score
            best_answer = answer

    if best_answer is None:
        return "Answer not found the questions not available in files"

    return best_answer


# =========================
# CHAT LOOP
# =========================
while True:
    q = input("\nAsk your question (type 'exit' to stop): ")

    if q.lower().strip() == "exit":
        print("Goodbye")
        break

    print("\nAnswer:")
    print(answer_question(q))