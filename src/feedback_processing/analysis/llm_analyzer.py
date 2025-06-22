# src/feedback_processing/analysis/llm_analyzer.py
import boto3
import json
from typing import List, Dict
from feedback_processing.analysis.sentiment_analyzer import analyze_sentiment
from feedback_processing.analysis.keyword_analyzer import analyze_keywords

bedrock = boto3.client(service_name='bedrock-runtime', region_name='eu-central-1') 

def analyze_feedback_with_llm(feedback_text: str) -> Dict:
    """
    Analyzes the interview feedback text using a Bedrock LLM,
    incorporating sentiment and keyword analysis.

    Args:
        feedback_text: The text content extracted from the feedback file.

    Returns:
        A dictionary containing the LLM analysis, sentiment, and keywords.
    """
    model_id = 'anthropic.claude-3-haiku-20240307-v1:0'  

    sentiment_result = analyze_sentiment(feedback_text)
    keyword_result = analyze_keywords(feedback_text)

    prompt = f"""Human: You are an expert in analyzing interview feedback for technical roles.
    Here is the sentiment analysis of the feedback: {sentiment_result}.
    Here are the top keywords identified: {keyword_result['top_keywords']}.

    Please read the following feedback and provide a structured analysis, including:

    - A brief summary of the overall feedback.
    - Key strengths of the candidate mentioned.
    - Areas where the candidate could improve.
    - Any specific examples or comments that highlight the candidate's performance.

    Feedback:
    {feedback_text}

    Assistant:"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1000,  # Adjust as needed
        "temperature": 0.7,  # Adjust for creativity vs. consistency
        "top_p": 0.9
    })

    accept = 'application/json'
    content_type = 'application/json'

    try:
        response = bedrock.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)
        response_body = json.loads(response.get('body').read())
        llm_analysis = response_body['content'][0]['text']
        return {"llm_analysis": llm_analysis, "sentiment": sentiment_result, "keywords": keyword_result['top_keywords']}
    except Exception as e:
        print(f"Error during Bedrock invocation for feedback analysis: {e}")
        return {"llm_analysis_error": str(e), "sentiment_error": sentiment_result.get("sentiment_polarity_error"), "keywords_error": keyword_result.get("top_keywords_error")}

if __name__ == "__main__":
    # Example usage with dummy feedback
    sample_feedback = """The candidate demonstrated strong technical skills in Python and data structures. They were able to solve the coding problem effectively. However, their communication skills could be improved, as they struggled to clearly articulate their thought process. The interviewer noted that the candidate was enthusiastic and eager to learn."""
    analysis_result = analyze_feedback_with_llm(sample_feedback)
    print(json.dumps(analysis_result, indent=4)) 