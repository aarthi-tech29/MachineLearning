# ==========================================
# RESEARCH PAPER CATEGORIZATION
# USING HIERARCHICAL CLUSTERING
# ==========================================

# IMPORT LIBRARIES
import os
import pandas as pd
import matplotlib.pyplot as plt

# os → Access files and folders
# pandas → Display similarity matrix
# matplotlib → Draw dendrogram graph

from PyPDF2 import PdfReader

# Reads PDF files
# Extracts text from research papers

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# TfidfVectorizer converts text into numerical values
# cosine_similarity measures similarity between papers

from scipy.cluster.hierarchy import linkage, dendrogram
# Performs Hierarchical Clustering
# Creates dendrogram visualization

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Used for NLP preprocessing
# Tokenization and stopword removal

# DOWNLOAD NLTK FILES
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
# Downloads tokenizer files
# Downloads stopword dataset

# ==========================================
# PDF TEXT EXTRACTION
# ==========================================

folder_path = "research_papers"

paper_names = []
paper_texts = []
# Stores PDF names
# Stores extracted text
for file in os.listdir(folder_path):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(folder_path, file)

        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:
            text += page.extract_text()

        paper_names.append(file)
        paper_texts.append(text)

print("PDF Text Extraction Completed")

# ==========================================
# NLP PREPROCESSING
# ==========================================

stop_words = set(stopwords.words('english'))

cleaned_texts = []

for text in paper_texts:

    text = text.lower()

    words = word_tokenize(text) # Splits text into words

    filtered_words = []

    for word in words:

        if word.isalpha() and word not in stop_words:
            filtered_words.append(word)

            # Removes:numbers,symbols,stopwords

    cleaned_text = " ".join(filtered_words)

    cleaned_texts.append(cleaned_text)

print("NLP Preprocessing Completed")

# ==========================================
# TF-IDF FEATURE EXTRACTION
# ==========================================

vectorizer = TfidfVectorizer()

# Converts text into numerical vectors
# TF-IDF identifies important words in each paper.

X = vectorizer.fit_transform(cleaned_texts)

# ==========================================
# TOPIC SIMILARITY ANALYSIS
# ==========================================

similarity_matrix = cosine_similarity(X)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=paper_names,
    columns=paper_names
)
# Calculates similarity between papers
# Values:
# 1.0 → highly similar
# 0.0 → unrelated

print("\nTopic Similarity Matrix")
print(similarity_df)

print("Topic Similarity Analysis Completed")

# ==========================================
# HIERARCHICAL CLUSTERING
# ==========================================

linkage_matrix = linkage(X.toarray(), method='ward')
# Displays similarity matrix in table format
# ==========================================
# DENDROGRAM VISUALIZATION
# ==========================================

plt.figure(figsize=(5,7))

plt.title("Research Paper Categorization Dendrogram")

dendrogram(
    linkage_matrix,
    labels=paper_names,
    leaf_rotation=90
)

plt.xlabel("Research Papers")
plt.ylabel("Distance")

plt.tight_layout()
plt.show()

print("Dendrogram Visualization Displayed")

# ==========================================
# PROJECT COMPLETED
# ==========================================

print("\nResearch Paper Categorization Completed Successfully")

# The model learns to group similar research papers based on their content
# The dendrogram visually shows how papers are clustered together based on their similarity
# Papers that are more similar will be grouped together at lower distances on the dendrogram
# This helps in categorizing research papers into different topics or themes based on their content
# The similarity matrix provides a numerical representation of how closely related each pair of papers is, which can be used for further analysis or to validate the clustering results
# Overall, this project demonstrates how hierarchical clustering can be applied to categorize research papers based on their textual content, providing insights into the relationships between different papers and helping researchers identify relevant literature in their field of study.
