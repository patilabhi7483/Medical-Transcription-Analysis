from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
import string
import math
import re
import random
from collections import defaultdict, Counter
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import CFG
from nltk.parse import ChartParser
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# =============================
# LOAD AND PREPARE DATA
# =============================
print("Loading NLP Pipeline...")
data = pd.read_csv("mtsamples.csv")

# Fixed text column (FOR MTSAMPLES)
if "transcription" in data.columns:
    data["text"] = data["transcription"].astype(str)
elif "description" in data.columns:
    data["text"] = data["description"].astype(str)
else:
    raise Exception("No usable text column found!")

# Preprocessing
stop_words = set(stopwords.words("english"))

def preprocess(text):
    if pd.isna(text):
        return []
    tokens = nltk.word_tokenize(str(text).lower())
    return [w for w in tokens if w not in stop_words and w not in string.punctuation]

data["tokens"] = data["text"].apply(preprocess)

# =============================
# LABEL CREATION
# =============================
def label(text):
    if pd.isna(text):
        return "general"
    text = str(text).lower()
    if "heart" in text or "cardiac" in text:
        return "cardiology"
    elif "brain" in text or "neurology" in text or "neural" in text:
        return "neurology"
    else:
        return "general"

data["label"] = data["text"].apply(label)

# =============================
# LANGUAGE MODEL (Bigram)
# =============================
sentences = data["text"].tolist()
random.shuffle(sentences)

train = sentences[:int(0.8*len(sentences))]
test = sentences[int(0.8*len(sentences)):]

tokens = []
for s in train:
    if isinstance(s, str):
        tokens.extend(nltk.word_tokenize(s.lower()))

unigram = Counter(tokens)
bigram = defaultdict(lambda: defaultdict(int))

for i in range(len(tokens)-1):
    bigram[tokens[i]][tokens[i+1]] += 1

V = len(set(tokens))

def prob(w1, w2):
    return (bigram[w1][w2] + 1) / (unigram[w1] + V)

def calculate_perplexity():
    log_prob = 0
    N = 0
    for s in test[:50]:
        if isinstance(s, str):
            words = nltk.word_tokenize(s.lower())
            N += len(words)
            for i in range(len(words)-1):
                log_prob += math.log(prob(words[i], words[i+1]))
    return math.exp(-log_prob / N) if N > 0 else 0

# Calculate perplexity once
perplexity_value = calculate_perplexity()

# =============================
# NAIVE BAYES CLASSIFIER
# =============================
class_counts = defaultdict(int)
word_counts = defaultdict(lambda: defaultdict(int))
total_words = defaultdict(int)
vocab = set()

for _, row in data.iterrows():
    c = row["label"]
    class_counts[c] += 1
    for w in row["tokens"]:
        word_counts[c][w] += 1
        total_words[c] += 1
        vocab.add(w)

V_vocab = len(vocab)
total_docs = len(data)

def word_prob(w, c):
    return (word_counts[c][w] + 1) / (total_words[c] + V_vocab)

def predict(text):
    words = preprocess(text)
    scores = {}
    for c in class_counts:
        score = math.log(class_counts[c] / total_docs)
        for w in words:
            score += math.log(word_prob(w, c))
        scores[c] = score
    return max(scores, key=scores.get)

# =============================
# EVALUATION METRICS
# =============================
sample_data = data.sample(min(100, len(data)), random_state=42)
y_true = []
y_pred = []

for _, r in sample_data.iterrows():
    y_true.append(r["label"])
    y_pred.append(predict(r["text"]))

accuracy_value = accuracy_score(y_true, y_pred)
f1_value = f1_score(y_true, y_pred, average='macro')
cm = confusion_matrix(y_true, y_pred)

# =============================
# INVERTED INDEX & TF-IDF
# =============================
index = defaultdict(set)

for i, row in data.iterrows():
    for w in row["tokens"]:
        index[w].add(i)

N = len(data)
df = {w: len(index[w]) for w in index}

def tf_idf_score(q, doc_idx):
    words = preprocess(q)
    doc_words = data.iloc[doc_idx]["tokens"]
    if len(doc_words) == 0:
        return 0
    s = 0
    for w in words:
        tf = doc_words.count(w) / len(doc_words)
        idf = math.log(N / (df.get(w, 1) + 1))
        s += tf * idf
    return s

def ranked(q, top_n=3):
    scores = [(i, tf_idf_score(q, i)) for i in range(len(data))]
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_n]

# =============================
# WORDNET QUERY EXPANSION
# =============================
def expand(q):
    ex = set(q.lower().split())
    for w in q.lower().split():
        for syn in wordnet.synsets(w):
            for l in syn.lemmas():
                ex.add(l.name().replace('_', ' '))
    return list(ex)[:10]  # Limit to 10 terms

# =============================
# FLASK ROUTES
# =============================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        req_data = request.get_json()
        text = req_data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Get predictions
        category = predict(text)
        
        # Get expanded terms
        expanded_terms = expand(text)
        
        # Get top retrieved documents
        top_results = ranked(text, top_n=3)
        results = []
        for doc_idx, score in top_results:
            if score > 0:
                doc_text = data.iloc[doc_idx]["text"]
                preview = doc_text[:150] + "..." if len(doc_text) > 150 else doc_text
                results.append({
                    'doc_id': int(doc_idx),
                    'score': round(score, 4),
                    'preview': preview
                })
        
        # Return response
        response = {
            'category': category,
            'expanded_terms': expanded_terms,
            'results': results,
            'metrics': {
                'accuracy': round(accuracy_value, 2),
                'f1': round(f1_value, 2),
                'perplexity': round(perplexity_value, 2)
            },
            'confusion_matrix': cm.tolist()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        'accuracy': round(accuracy_value, 2),
        'f1': round(f1_value, 2),
        'perplexity': round(perplexity_value, 2),
        'confusion_matrix': cm.tolist()
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Accuracy: {accuracy_value:.2f}, F1: {f1_value:.2f}, Perplexity: {perplexity_value:.2f}")
    app.run(debug=False, host='0.0.0.0', port=5000)
