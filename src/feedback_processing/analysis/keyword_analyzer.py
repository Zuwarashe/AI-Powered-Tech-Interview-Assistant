# src/feedback_processing/analysis/keyword_analyzer.py
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Dict

try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')

def analyze_keywords(text: str, top_n: int = 10) -> Dict:
    """
    Identifies the top N keywords in the given text.

    Args:
        text: The text to analyze.
        top_n: The number of top keywords to return.

    Returns:
        A dictionary containing the top N keywords and their counts.
    """
    try:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        filtered_words = [w for w in word_tokens if not w in stop_words and re.match(r'^[a-zA-Z]+$', w)]
        keyword_counts = Counter(filtered_words)
        top_keywords = keyword_counts.most_common(top_n)
        return {"top_keywords": top_keywords}
    except Exception as e:
        print(f"Error during keyword analysis: {e}")
        return {"top_keywords_error": str(e)}

if __name__ == "__main__":
    sample_text = "The candidate demonstrated strong technical skills, especially in Python and data structures. Python was mentioned multiple times. The candidate also showed enthusiasm."
    keywords = analyze_keywords(sample_text)
    print(f"Top Keywords: {keywords}")