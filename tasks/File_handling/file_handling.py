import pdfplumber
from docx import Document
import pandas as pd
import re

# pdfplumber → read PDF text cleanly
# Document → read Word (.docx)
# pandas → read CSV
# re → clean text using patterns

# ---------- SYNONYMS ----------
SYNONYMS = {
    "consumed": "used",
    "consume": "used",
    "use": "used",
    "usage": "used",
    "purpose": "used",
    "reason": "used",
    "application": "used",
    "applications": "used",
    "function": "used",
    "functions": "used",
    "benefit": "used",
    "benefits": "used",

    "explain": "what",
    "describe": "what",
    "tell": "what",

    "repeat": "loop",
    "iteration": "loop"
}

# ---------- QUESTION DETECTION ----------
QUESTION_WORDS = (
    "what", "why", "how", "when", "where",
    "explain", "define", "describe", "tell"
)

def is_question(line):
    line_lower = line.lower().strip()

    if "?" in line_lower:
        return True

    if any(line_lower.startswith(word) for word in QUESTION_WORDS):
        return True

    if "meaning" in line_lower or "definition" in line_lower: # Detect special cases
        return True

    return False


# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = re.sub(r'^\d+\.\s*', '', text) # Removes numbering
    text = re.sub(r'[^\w\s]', '', text) # Removes punctuation
    return text.lower().strip()


# ---------- EXTRACT Q-A ----------
def extract_qa(lines):
    qa_pairs = []
    current_q = None
    current_a = []

    for line in lines:
        line = line.strip()
        if not line: # skip empty lines
            continue

        if is_question(line): # Detect Question
            if current_q and current_a:
                qa_pairs.append((current_q, " ".join(current_a))) # Save previous Q-A pair

            current_q = clean_text(line) # Start new question
            current_a = []
        else:
            if current_q:
                current_a.append(line)

    if current_q and current_a:
        qa_pairs.append((current_q, " ".join(current_a))) # Save last Q&A

    return qa_pairs


# ---------- READ PDF ----------
def read_pdf(file_path):
    lines = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.split("\n"))
    return extract_qa(lines)

# Read page text
# Split into lines
# Send to extract_qa()


# ---------- READ DOCX ----------
def read_docx(file_path):
    doc = Document(file_path)
    lines = [para.text for para in doc.paragraphs]
    return extract_qa(lines)


# ---------- READ CSV ----------
def read_csv(file_path):
    df = pd.read_csv(file_path)
    qa_pairs = []

    for _, row in df.iterrows():
        q = clean_text(str(row.iloc[0]))
        a = str(row.iloc[1])
        qa_pairs.append((q, a))

    return qa_pairs


# ---------- LOAD FILES ----------
qa_data = []
qa_data.extend(read_pdf("data.pdf"))
qa_data.extend(read_docx("data.docx"))
qa_data.extend(read_csv("data.csv"))

# Combine everything

print(f"Loaded {len(qa_data)} Q&A pairs.")


# ---------- SMART SEARCH ----------
def answer_question(user_question):
    user_question = clean_text(user_question)

    stop_words = {
        "what", "is", "the", "about", "explain",
        "tell", "me", "define", "why", "for"
    }
    # Removes useless words
    words = []
    for w in user_question.split():
        if w not in stop_words:
            w = SYNONYMS.get(w, w)
            words.append(w)

    best_answer = ""
    max_score = 0

    for question, answer in qa_data: # Compare with stored questions
        q_words = [SYNONYMS.get(w, w) for w in question.split()] # Normalize stored question

        score = 0

        for word in words:
            if word in q_words:
                score += 2 # More matches → higher score

        if user_question in question:
            score += 5 # Exact match bonus

        if score > max_score:
            max_score = score
            best_answer = answer

    if max_score >= 2:
        return best_answer
    else:
        return "Answer not found in document."


# ---------- TERMINAL LOOP ----------
while True:
    q = input("\nAsk your question (type 'exit' to stop): ")

    if q.lower() == "exit":
        print("Goodbye")
        break

    ans = answer_question(q)

    if ans != "Answer not found in document.":
        print("\nAnswer:")
        print(ans)
    else:
        print("\nAnswer not found in document.")

# User Question
#       ↓
# Clean text
#       ↓
# Remove stop words
#       ↓
# Apply synonyms
#       ↓
# Compare with stored Q&A
#       ↓
# Calculate score
#       ↓
# Pick best match
#       ↓
# Print answer


