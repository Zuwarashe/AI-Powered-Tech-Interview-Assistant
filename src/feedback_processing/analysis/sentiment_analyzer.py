# src/feedback_processing/analysis/sentiment_analyzer.py
from textblob import TextBlob
from typing import Dict

def analyze_sentiment(text: str) -> Dict:
    """
    Analyzes the sentiment of the given text.

    Args:
        text: The text to analyze.

    Returns:
        A dictionary containing the sentiment polarity and subjectivity.
        Polarity ranges from -1 (negative) to 1 (positive).
        Subjectivity ranges from 0 (objective) to 1 (subjective).
    """
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return {"sentiment_polarity": sentiment.polarity, "sentiment_subjectivity": sentiment.subjectivity}
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return {"sentiment_polarity_error": str(e)}

if __name__ == "__main__":
    sample_text_positive = "The candidate was very enthusiastic and performed well."
    sentiment_positive = analyze_sentiment(sample_text_positive)
    print(f"Sentiment (Positive): {sentiment_positive}")

    sample_text_negative = "The candidate lacked the required technical skills."
    sentiment_negative = analyze_sentiment(sample_text_negative)
    print(f"Sentiment (Negative): {sentiment_negative}")

    sample_text_neutral = "The interviewer asked several questions."
    sentiment_neutral = analyze_sentiment(sample_text_neutral)
    print(f"Sentiment (Neutral): {sentiment_neutral}")
