import uuid
import json
import boto3
from langchain_core.documents import Document
from src.data_automation.pipelines.dynamodb_operations import DynamoDBHandler
from src.resume_processing.information_extraction import embed_text
from typing import Dict, Any, List, Optional

BEDROCK_REGION = "eu-central-1"
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

bedrock = boto3.client(service_name='bedrock-runtime', region_name=BEDROCK_REGION)
dynamodb_handler = DynamoDBHandler(table_name="ResumeJobMatcher")

def extract_job_details_llm(doc: Document) -> Optional[List[Dict[str, Any]]]:
    """Uses Claude 3 via Bedrock to extract structured job details from text for each level"""
    
    # Check if we already have this JD processed
    if doc.metadata.get('source'):
        existing_jds = [jd for jd in dynamodb_handler.get_all_job_descriptions() 
                      if jd.get('metadata', {}).get('source') == doc.metadata['source']]
        if existing_jds:
            return existing_jds

    prompt = f"""
You are an expert at extracting structured information from career path documents. Given the raw text below, 
extract the information for each career level (Junior, Mid-Level, Senior, Principal) as separate JSON objects.

For each level, return the information in this structure:
{{
  "level": string (e.g., "Junior", "Mid-Level", etc.),
  "title": string,
  "experience": string (years or range),
  "focus": string,
  "core_requirements": [string],
  "soft_skills": [string],
  "technologies_mentioned": [string]
}}

Input Text:
\"\"\"{doc.page_content}\"\"\"

Important Notes:
- Extract details for each level separately (Junior, Mid-Level, Senior, Principal)
- Include the emoji/color indicator (🟢, 🔵, 🔴, ⚫) in the title
- Extract exact years/ranges for experience (e.g. "2-5 years")
- List all technologies/tools mentioned for each level
- Preserve the original wording when possible
- Return a list of JSON objects, one for each level found
"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000,
        "temperature": 0.3
    })

    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=MODEL_ID,
            accept='application/json',
            contentType='application/json'
        )

        response_body = json.loads(response.get("body").read())
        content_list = response_body.get("content", [])
        content = "".join([item.get("text", "") for item in content_list]).strip()

        if not content:
            print("⚠️ LLM returned empty content.")
            return None

        # Extract JSON from response - looking for a list of objects
        json_start = content.find("[")
        json_end = content.rfind("]")
        if json_start == -1 or json_end == -1:
            print("⚠️ JSON array not found in content.")
            print("Raw content:\n", content[:500])
            return None

        json_text = content[json_start:json_end+1]

        try:
            extracted_levels = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding failed: {e}")
            print("Raw content:\n", content[:500])
            return None

        if not extracted_levels or not isinstance(extracted_levels, list):
            print("⚠️ No valid levels extracted.")
            return None

        # Prepare and save each level
        saved_levels = []
        for level_data in extracted_levels:
            if not isinstance(level_data, dict):
                continue
                
            # Create embedding text from key fields
            embedding_text = (
                f"Title: {level_data.get('title', '')}\n"
                f"Level: {level_data.get('level', '')}\n"
                f"Experience: {level_data.get('experience', '')}\n"
                f"Focus: {level_data.get('focus', '')}\n"
                f"Requirements: {', '.join(level_data.get('core_requirements', []))}\n"
                f"Skills: {', '.join(level_data.get('soft_skills', []))}\n"
                f"Technologies: {', '.join(level_data.get('technologies_mentioned', []))}"
            )
            
            # Generate embedding for this specific level
            embedding = embed_text(embedding_text)
            
            # Add metadata and other fields
            level_data.update({
                "metadata": doc.metadata,
                "full_text": doc.page_content[:500] + "...",
                "id": str(uuid.uuid4()),
                "type": "job_description",
                "embedding": embedding 
            })
            
            # Save to DynamoDB
            if dynamodb_handler.save_job_description(level_data):
                saved_levels.append(level_data)
            else:
                print(f"⚠️ Failed to save level {level_data.get('level')} to DynamoDB")

        return saved_levels if saved_levels else None

    except Exception as e:
        print(f"💥 Error extracting job details with LLM: {e}")
        return None