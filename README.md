# Medical Text Classification and Information Retrieval System

A web-based NLP application for medical text classification using Naive Bayes and TF-IDF information retrieval.

## Features

- **Medical Text Classification**: Classifies text into Cardiology, Neurology, or General categories
- **Query Expansion**: Uses WordNet to expand search terms
- **TF-IDF Information Retrieval**: Retrieves relevant medical documents
- **NLP Metrics**: Displays Perplexity, Accuracy, and F1 Score
- **Confusion Matrix**: Visual representation of classification performance

## Tech Stack

- **Backend**: Flask (Python)
- **NLP Libraries**: NLTK, scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript
- **UI Icons**: FontAwesome

## File Structure

```
NLP_Project/
├── app.py                 # Flask backend with NLP pipeline
├── code.py                # Original NLP pipeline script
├── mtsamples.csv          # Medical transcription dataset
├── templates/
│   └── index.html         # Main HTML template
├── static/
│   ├── style.css          # CSS styling
│   └── script.js          # Frontend JavaScript
└── requirements.txt       # Python dependencies
```

## Installation

1. Install required packages:
```bash
pip install flask nltk pandas numpy scikit-learn
```

2. Download NLTK data (handled automatically on first run)

## Running the Application

Start the Flask server:

```bash
python app.py
```

The application will be available at: **http://127.0.0.1:5000**

## API Endpoint

### POST /analyze

**Request:**
```json
{
  "text": "patient has heart disease and chest pain"
}
```

**Response:**
```json
{
  "category": "cardiology",
  "expanded_terms": ["disease", "illness", "condition"],
  "results": [
    {
      "doc_id": 120,
      "score": 0.087,
      "preview": "patient suffers from..."
    }
  ],
  "metrics": {
    "accuracy": 0.89,
    "f1": 0.83,
    "perplexity": 928.14
  },
  "confusion_matrix": [[30, 5, 2], [4, 25, 3], [3, 2, 26]]
}
```

## Usage

1. Open the application in your browser
2. Enter medical symptoms, diagnosis, or healthcare text in the textarea
3. Click "Analyze Text" button
4. View the results including:
   - Predicted medical category
   - Expanded query terms
   - Top retrieved documents
   - NLP performance metrics
   - Confusion matrix

## Design

- Clean, modern student-project style UI
- Medical/healthcare theme with blue, white, and cyan colors
- Responsive design for desktop and mobile
- Smooth fade-in animations
- Card-based layout with gradient backgrounds
