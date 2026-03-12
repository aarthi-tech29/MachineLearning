# =============================Live news detection==========================================
import requests
from dotenv import load_dotenv
import os

load_dotenv()


# ===============================
# API KEYS
# ===============================
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")

# HuggingFace Model
HF_API_URL = "https://router.huggingface.co/hf-inference/models/hamzab/roberta-fake-news-classification"

HF_HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
}

# ===============================
# TRUSTED NEWS SOURCES
# ===============================
trusted_sources = [
    "the-washington-post","washington post","cnn","bbc","bbc-news",
    "reuters","npr","espn","associated press","detroit free press",
    "el-balad.com","cbs sports","space.com","the seattle times","yahoo"
]

# ===============================
# GET LIVE NEWS
# ===============================
def get_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=5&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "articles" in data:
            return data["articles"]
        return []
    except:
        print("Error fetching news")
        return []

# ===============================
# ML FAKE NEWS DETECTION
# ===============================
def ml_prediction(text):
    payload = {"inputs": text}
    try:
        response = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json=payload
        )
        result = response.json()
        predictions = result[0]
        best = max(predictions, key=lambda x: x["score"])
        label = best["label"]
        score = best["score"]
        return label, score
    except:
        return "LABEL_0", 0

# ===============================
# CONVERT LABEL
# ===============================
def convert_label(label):
    label = label.lower()
    if label == "label_1":
        return "FAKE NEWS ❌"
    elif label == "label_0":
        return "REAL NEWS ✅"
    elif "fake" in label:
        return "FAKE NEWS ❌"
    elif "real" in label or "true" in label:
        return "REAL NEWS ✅"
    else:
        return "REAL NEWS ✅"

# ===============================
# FACT CHECK API
# ===============================
def fact_check(query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={FACTCHECK_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "claims" in data and len(data["claims"]) > 0:
            claim = data["claims"][0]
            publisher = claim["claimReview"][0]["publisher"]["name"]
            rating = claim["claimReview"][0]["textualRating"]
            return True, publisher, rating
        else:
            return False, None, None
    except:
        return False, None, None

# ===============================
# FINAL DECISION LOGIC
# ===============================
def final_decision(source, prediction):
    source_lower = source.lower()
    if source_lower in trusted_sources and prediction == "FAKE NEWS ❌":
        return "LIKELY REAL (Trusted Source)"
    elif source_lower in trusted_sources and prediction == "REAL NEWS ✅":
        return "REAL NEWS"
    elif source_lower not in trusted_sources and prediction == "FAKE NEWS ❌":
        return "FAKE NEWS"
    else:
        return "UNVERIFIED"

# ===============================
# IMPROVED TRUST SCORE CALCULATION
# ===============================
def calculate_trust_score_v2(source, ml_score, prediction):
    source_lower = source.lower()

    # Source trust
    if source_lower in trusted_sources:
        source_trust = 95
    else:
        source_trust = 50

    # ML reliability
    if prediction == "REAL NEWS ✅":
        ml_trust = ml_score * 100
    else:  # FAKE NEWS
        ml_trust = (1 - ml_score) * 100

    # Weighted combination: 40% source, 60% ML
    final_trust = 0.4 * source_trust + 0.6 * ml_trust

    return round(source_trust, 2), round(ml_trust, 2), round(final_trust, 2)

# ===============================
# MAIN PROGRAM
# ===============================
print("\nFetching Live News...\n")
articles = get_news()

for article in articles:
    title = article["title"] or ""
    description = article["description"] or ""
    source = article["source"].get("id") or article["source"].get("name", "Unknown")
    text = title + " " + description

    # ML Prediction
    label, score = ml_prediction(text)
    prediction = convert_label(label)

    # Fact check
    fact_found, publisher, rating = fact_check(title)

    # Final decision
    final_result = final_decision(source, prediction)

    # Improved Trust Score
    source_trust, ml_trust, trust_score = calculate_trust_score_v2(source, score, prediction)

    # ===============================
    # PRINT RESULTS
    # ===============================
    print("News:", title)
    print("Source:", source)

    if fact_found:
        print("Fact Check:", rating, "by", publisher)
    else:
        if source.lower() in trusted_sources:
            print("Fact Check: Trusted News Source ✅")
        else:
            print("Fact Check: Not verified ⚠️")

    print("ML Prediction:", prediction)
    print("Confidence:", round(score * 100, 2), "%")

    print("Source Trust:", source_trust, "%")
    print("ML Trust:", ml_trust, "%")
    print("Final Trust Score:", trust_score, "%")

    print("Final Verdict:", final_result)
    print("-" * 60)

# ==============================Hybrid both rule based and live based detection==========================================
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# ===============================
# API KEYS
# ===============================
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")

# HuggingFace model
HF_API_URL = "https://router.huggingface.co/hf-inference/models/hamzab/roberta-fake-news-classification"
HF_HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Trusted news sources
trusted_sources = [
    "the-washington-post","washington post","cnn","bbc","bbc-news",
    "reuters","npr","espn","associated press","detroit free press",
    "el-balad.com","cbs sports","space.com","the seattle times","yahoo"
]

# ===============================
# ML Prediction
# ===============================
def ml_prediction(text):
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
        result = response.json()
        predictions = result[0]
        best = max(predictions, key=lambda x: x["score"])
        label = best["label"]
        score = best["score"]
        if label.lower() == "label_1" or "fake" in label.lower():
            return "FAKE NEWS ❌", score
        else:
            return "REAL NEWS ✅", score
    except:
        return "REAL NEWS ✅", 0.0

# ===============================
# Rule-based check
# ===============================
def rule_based_check(text):
    suspicious_keywords = [
        "aliens","mars next year","landed yesterday","time travel",
        "miracle cure","vaccine kills","magic potion","secret lab"
    ]
    return any(word.lower() in text.lower() for word in suspicious_keywords)

# ===============================
# Fact Check API
# ===============================
def fact_check(text):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={text}&key={FACTCHECK_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "claims" in data and len(data["claims"]) > 0:
            claim = data["claims"][0]
            publisher = claim["claimReview"][0]["publisher"]["name"]
            rating = claim["claimReview"][0]["textualRating"]
            return True, publisher, rating
        else:
            return False, None, None
    except:
        return False, None, None

# ===============================
# Weighted Trust Score
# ===============================
def calculate_trust_score(source, ml_score, prediction):
    source_trust = 95 if source.lower() in trusted_sources else 50
    ml_trust = ml_score*100 if "REAL" in prediction else (1-ml_score)*100
    final_trust = 0.4*source_trust + 0.6*ml_trust
    return round(source_trust,2), round(ml_trust,2), round(final_trust,2)

# ===============================
# Final Verdict
# ===============================
def final_decision(source, prediction):
    s = source.lower()
    if s in trusted_sources and "FAKE" in prediction:
        return "LIKELY REAL (Trusted Source)"
    elif s in trusted_sources and "REAL" in prediction:
        return "REAL NEWS"
    elif s not in trusted_sources and "FAKE" in prediction:
        return "FAKE NEWS"
    else:
        return "UNVERIFIED"

# ===============================
# Detect news (single article)
# ===============================
def detect_news(text, source="Unknown"):
    if rule_based_check(text):
        prediction = "FAKE NEWS ❌ (Suspicious keywords detected)"
        ml_score = 0.999
    else:
        prediction, ml_score = ml_prediction(text)

    fact_found, publisher, rating = fact_check(text)
    source_trust, ml_trust, final_trust = calculate_trust_score(source, ml_score, prediction)
    final_result = final_decision(source, prediction)

    print("\nNews:", text)
    print("Source:", source)
    if fact_found:
        print("Fact Check:", rating, "by", publisher)
    else:
        print("Fact Check:", "Trusted News Source ✅" if source.lower() in trusted_sources else "Not verified ⚠️")
    print("ML Prediction:", prediction)
    print("Confidence:", round(ml_score*100,2), "%")
    print("Source Trust:", source_trust, "%")
    print("ML Trust:", ml_trust, "%")
    print("Final Trust Score:", final_trust, "%")
    print("Final Verdict:", final_result)
    print("-"*60)

# ===============================
# Fetch live news
# ===============================
def get_live_news(country="us", limit=5):
    url = f"https://newsapi.org/v2/top-headlines?country={country}&pageSize={limit}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        return data.get("articles", [])
    except:
        return []

# ===============================
# Main Program
# ===============================
print("\n=== Hybrid Fake News Detection System ===")
print("Type 'exit' to quit")
print("Type 'live' to fetch top live news\n")

while True:
    choice = input("Enter news text / command: ").strip()
    if choice.lower() == "exit":
        break
    elif choice.lower() == "live":
        articles = get_live_news()
        for article in articles:
            title = article.get("title","")
            desc = article.get("description","")
            source = article.get("source",{}).get("name","Unknown")
            text = f"{title} {desc}"
            detect_news(text, source)
    else:
        source_text = input("Enter news source (or leave blank for Unknown): ").strip() or "Unknown"
        detect_news(choice, source_text)


# Normal news (should detect REAL NEWS):
# “Government announced a new education policy today”
# “Apple releases the new iPhone 15 with advanced camera features”
# Suspicious / fake news (should detect FAKE NEWS):
# “Aliens landed in India yesterday”
# “Scientists confirm humans will live on Mars next year”
# “Miracle cure for all diseases discovered in Antarctica”
# “Time travel machine invented in secret lab”
# Clickbait-style or partially suspicious (ML will decide):
# “Maxx Crosby back on the market for Detroit Lions? Probably not”
# “Agents: Colts to re-sign QB Daniel Jones to 2-year, $88M deal”

# ===========================Rule based detection===============================================
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# ===============================
# HuggingFace Model for Fake News
# ===============================
API_URL = "https://router.huggingface.co/hf-inference/models/hamzab/roberta-fake-news-classification"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"
}

# ===============================
# Trusted Sources
# ===============================
trusted_sources = [
    "the-washington-post","washington post","cnn","bbc","bbc-news",
    "reuters","npr","espn","associated press","detroit free press",
    "el-balad.com","cbs sports","space.com","the seattle times","yahoo"
]

# ===============================
# Fact Check API Key
# ===============================
FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")

# ===============================
# ML Prediction
# ===============================
def ml_prediction(text):
    payload = {"inputs": text}
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        result = response.json()
        predictions = result[0]
        best_prediction = max(predictions, key=lambda x: x["score"])
        label = best_prediction["label"]
        score = best_prediction["score"]
        if label.lower() == "label_1" or "fake" in label.lower():
            label = "FAKE NEWS ❌"
        else:
            label = "REAL NEWS ✅"
        return label, score
    except Exception as e:
        print("ML Prediction Error:", e)
        return "REAL NEWS ✅", 0.0

# ===============================
# Fact Check API
# ===============================
def fact_check(text):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={text}&key={FACTCHECK_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "claims" in data and len(data["claims"]) > 0:
            claim = data["claims"][0]
            publisher = claim["claimReview"][0]["publisher"]["name"]
            rating = claim["claimReview"][0]["textualRating"]
            return True, publisher, rating
        else:
            return False, None, None
    except:
        return False, None, None

# ===============================
# Final Decision
# ===============================
def final_decision(source, prediction):
    source_lower = source.lower()
    if source_lower in trusted_sources and prediction == "FAKE NEWS ❌":
        return "LIKELY REAL (Trusted Source)"
    elif source_lower in trusted_sources and prediction == "REAL NEWS ✅":
        return "REAL NEWS"
    elif source_lower not in trusted_sources and prediction == "FAKE NEWS ❌":
        return "FAKE NEWS"
    else:
        return "UNVERIFIED"

# ===============================
# Weighted Trust Score
# ===============================
def calculate_trust_score(source, ml_score, prediction):
    source_lower = source.lower()
    source_trust = 95 if source_lower in trusted_sources else 50
    if prediction == "REAL NEWS ✅":
        ml_trust = ml_score * 100
    else:
        ml_trust = (1 - ml_score) * 100
    final_trust = 0.4 * source_trust + 0.6 * ml_trust
    return round(source_trust,2), round(ml_trust,2), round(final_trust,2)

# ===============================
# Rule-based check for sensational/fake claims
# ===============================
def rule_based_check(text):
    suspicious_keywords = [
        "aliens", "mars next year", "landed yesterday", "time travel", "miracle cure", "vaccine kills"
    ]
    if any(word.lower() in text.lower() for word in suspicious_keywords):
        return True
    return False

# ===============================
# Detect Fake News (Full System)
# ===============================
def detect_news(text, source="Unknown"):
    if rule_based_check(text):
        prediction = "FAKE NEWS ❌ (Suspicious keywords detected)"
        ml_score = 0.999
    else:
        prediction, ml_score = ml_prediction(text)

    fact_found, publisher, rating = fact_check(text)
    source_trust, ml_trust, final_trust = calculate_trust_score(source, ml_score, prediction)
    final_result = final_decision(source, prediction)

    print("\nNews:", text)
    print("Source:", source)
    if fact_found:
        print("Fact Check:", rating, "by", publisher)
    else:
        if source.lower() in trusted_sources:
            print("Fact Check: Trusted News Source ✅")
        else:
            print("Fact Check: Not verified ⚠️")
    print("ML Prediction:", prediction)
    print("Confidence:", round(ml_score*100,2), "%")
    print("Source Trust:", source_trust, "%")
    print("ML Trust:", ml_trust, "%")
    print("Final Trust Score:", final_trust, "%")
    print("Final Verdict:", final_result)
    print("-"*60)

# ===============================
# Terminal Input Loop
# ===============================
print("\n=== Fake News Detection System ===")
print("Type 'exit' to quit\n")

while True:
    news_text = input("Enter news text: ")
    if news_text.lower() == "exit":
        break
    source_text = input("Enter news source (or leave blank for Unknown): ")
    source_text = source_text.strip() if source_text else "Unknown"
    detect_news(news_text, source_text)

