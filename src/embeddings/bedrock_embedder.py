# src/embeddings/bedrock_embedder.py

import boto3
import os
import json

# Optional: Load from env vars or config
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v1")  # Change this if using another model

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

def embed_text(text):
    """
    Embeds a single string using Bedrock Titan model.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string.")

    client = get_bedrock_client()
    body = {
        "inputText": text
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json"
    )

    response_body = json.loads(response["body"].read())
    return response_body.get("embedding", [])

def embed_batch(text_list):
    """
    Embeds a list of strings using Bedrock Titan model.
    """
    if not isinstance(text_list, list):
        raise ValueError("Input must be a list of strings.")

    return [embed_text(text) for text in text_list]
