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
import nltk

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# -----------------------------
# DOWNLOAD NLTK (run once)
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

print("===== NLP COMPLETE PIPELINE (MTSAMPLES) =====")

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("mtsamples.csv")

print("Columns:", data.columns)

# ✅ FIXED TEXT COLUMN (FOR MTSAMPLES)
if "transcription" in data.columns:
    data["text"] = data["transcription"].astype(str)

elif "description" in data.columns:
    data["text"] = data["description"].astype(str)

else:
    raise Exception("No usable text column found!")

# -----------------------------
# STEP 1: PREPROCESSING
# -----------------------------
print("\n--- PREPROCESSING ---")

stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return [w for w in tokens if w not in stop_words and w not in string.punctuation]

data["tokens"] = data["text"].apply(preprocess)

print(data["tokens"].head(3))


# -----------------------------
# STEP 2: LANGUAGE MODEL
# -----------------------------
print("\n--- LANGUAGE MODEL ---")

sentences = data["text"].tolist()
random.shuffle(sentences)

train = sentences[:int(0.8*len(sentences))]
test = sentences[int(0.8*len(sentences)):]

tokens=[]
for s in train:
    tokens.extend(nltk.word_tokenize(s.lower()))

unigram = Counter(tokens)
bigram = defaultdict(lambda: defaultdict(int))

for i in range(len(tokens)-1):
    bigram[tokens[i]][tokens[i+1]] += 1

V=len(set(tokens))

def prob(w1,w2):
    return (bigram[w1][w2]+1)/(unigram[w1]+V)

def perplexity():
    log_prob=0
    N=0
    for s in test[:50]:
        words=nltk.word_tokenize(s.lower())
        N+=len(words)
        for i in range(len(words)-1):
            log_prob+=math.log(prob(words[i],words[i+1]))
    return math.exp(-log_prob/N)

print("Perplexity:", perplexity())


# -----------------------------
# STEP 3: REGEX + POS
# -----------------------------
print("\n--- WORD ANALYSIS ---")

sample = data["text"].iloc[0]

print("Numbers:", re.findall(r'\d+', sample))
print("Capital Words:", re.findall(r'\b[A-Z][a-z]+\b', sample))
print("ING Words:", re.findall(r'\b\w+ing\b', sample))

print("POS:", nltk.pos_tag(nltk.word_tokenize(sample))[:10])


# -----------------------------
# STEP 4: STEMMING + LEMMATIZATION
# -----------------------------
print("\n--- STEMMING VS LEMMATIZATION ---")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running","studies","playing","diagnosed"]

for w in words:
    print(w, "->", stemmer.stem(w), "/", lemmatizer.lemmatize(w))


# -----------------------------
# STEP 5: CFG
# -----------------------------
print("\n--- CFG PARSER ---")

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP | V
Det -> 'the'
N -> 'patient' | 'doctor'
V -> 'treats' | 'examines'
""")

parser = ChartParser(grammar)

for tree in parser.parse("the doctor examines".split()):
    print(tree)


# -----------------------------
# STEP 6: LABEL CREATION (FIXED FOR MEDICAL)
# -----------------------------
print("\n--- LABEL CREATION ---")

def label(text):
    text=text.lower()
    if "heart" in text or "cardiac" in text:
        return "cardiology"
    elif "brain" in text or "neurology" in text:
        return "neurology"
    else:
        return "general"

data["label"]=data["text"].apply(label)

print(data["label"].value_counts())


# -----------------------------
# STEP 7: NAIVE BAYES
# -----------------------------
print("\n--- NAIVE BAYES ---")

class_counts=defaultdict(int)
word_counts=defaultdict(lambda: defaultdict(int))
total_words=defaultdict(int)
vocab=set()

for _,row in data.iterrows():
    c=row["label"]
    class_counts[c]+=1
    for w in row["tokens"]:
        word_counts[c][w]+=1
        total_words[c]+=1
        vocab.add(w)

V=len(vocab)
total_docs=len(data)

def word_prob(w,c):
    return (word_counts[c][w]+1)/(total_words[c]+V)

def predict(text):
    words=preprocess(text)
    scores={}
    for c in class_counts:
        score=math.log(class_counts[c]/total_docs)
        for w in words:
            score+=math.log(word_prob(w,c))
        scores[c]=score
    return max(scores,key=scores.get)

print("Prediction:", predict("patient has heart problem"))


# -----------------------------
# EVALUATION
# -----------------------------
sample_data = data.sample(100)

y_true=[]
y_pred=[]

for _,r in sample_data.iterrows():
    y_true.append(r["label"])
    y_pred.append(predict(r["text"]))

print("Accuracy:", accuracy_score(y_true,y_pred))
print("F1:", f1_score(y_true,y_pred,average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_true,y_pred))


# -----------------------------
# STEP 8: INVERTED INDEX
# -----------------------------
print("\n--- INVERTED INDEX ---")

index = defaultdict(set)

for i,row in data.iterrows():
    for w in row["tokens"]:
        index[w].add(i)

print("Sample:", list(index.items())[:3])


# -----------------------------
# BOOLEAN SEARCH
# -----------------------------
print("\n--- BOOLEAN SEARCH ---")

def boolean(q):
    t=q.lower().split()
    if "and" in t:
        return list(index[t[0]] & index[t[2]])
    elif "or" in t:
        return list(index[t[0]] | index[t[2]])

print("Query:", boolean("heart and patient"))


# -----------------------------
# TF-IDF
# -----------------------------
print("\n--- TF-IDF ---")

N=len(data)
df={w:len(index[w]) for w in index}

def score(q,doc):
    words=preprocess(q)
    doc_words=data.iloc[doc]["tokens"]
    s=0
    for w in words:
        tf=doc_words.count(w)/len(doc_words)
        idf=math.log(N/(df.get(w,1)))
        s+=tf*idf
    return s

def ranked(q):
    scores=[(i,score(q,i)) for i in range(len(data))]
    return sorted(scores,key=lambda x:x[1],reverse=True)

print("Top Results:", ranked("heart disease")[:3])


# -----------------------------
# WORDNET EXPANSION
# -----------------------------
print("\n--- WORDNET ---")

def expand(q):
    ex=set(q.split())
    for w in q.split():
        for syn in wordnet.synsets(w):
            for l in syn.lemmas():
                ex.add(l.name())
    return list(ex)

print("Expanded:", expand("disease")[:10])


# -----------------------------
# FINAL PIPELINE
# -----------------------------
print("\n===== FINAL DEMO =====")

user_input = input("Enter text: ")

print("Category:", predict(user_input))

results = ranked(user_input)

for doc,score in results[:3]:
    print("\nDoc:", doc, "Score:", round(score,3))