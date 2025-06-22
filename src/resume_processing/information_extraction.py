# src/resume_processing/information_extraction.py

import uuid
import json
import boto3
from langchain_core.documents import Document
from src.data_automation.pipelines.dynamodb_operations import DynamoDBHandler

BEDROCK_REGION = "eu-central-1"
MODEL_ID = "amazon.titan-embed-text-v1"

bedrock = boto3.client(service_name='bedrock-runtime', region_name=BEDROCK_REGION)

# Initialize DynamoDB handler
dynamodb_handler = DynamoDBHandler(table_name="ResumeJobMatcher")

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

def embed_text(text):
    """Calls Amazon Bedrock Titan model to generate an embedding for input text."""
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string.")

    client = get_bedrock_client()
    body = { "inputText": text }

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json"
    )

    response_body = json.loads(response["body"].read())
    return response_body.get("embedding", [])

def extract_profile_using_llm(text, metadata=None):
    """Uses Claude via Bedrock to extract profile data from raw resume text."""
    model_id = 'anthropic.claude-3-haiku-20240307-v1:0'

    prompt = f"""
You are an expert at extracting structured information from resumes. Given the raw resume text below, extract the following information and return it as a JSON object with the following structure:

{{
  "name": string,
  "contact": {{
    "email": string | null,
    "phone": string | null,
    "linkedin": string | null
  }},
  "education": [{{"degree": string, "major": string | null, "institution": string, "year": string | null}}],
  "experience": [{{"title": string, "company": string, "start_date": string | null, "end_date": string | null, "duration": string | null, "responsibilities": [string]}}],
  "projects": [{{"name": string, "description": string, "technologies": [string]}}],
  "skills": [string]
}}

Resume:
\"\"\"{text}\"\"\"
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
            modelId=model_id,
            accept='application/json',
            contentType='application/json'
        )

        response_body = json.loads(response.get("body").read())
        content_list = response_body.get("content", [])
        content = "".join([item.get("text", "") for item in content_list]).strip()

        if not content:
            print("⚠️ LLM returned empty content.")
            return None

        json_start = content.find("{")
        json_end = content.rfind("}")
        if json_start == -1 or json_end == -1:
            print("⚠️ JSON block not found in content.")
            print("Raw content:\n", content[:500])
            return None

        json_text = content[json_start:json_end+1]

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding failed: {e}")
            print("Raw content:\n", content[:500])
            return None

        parsed.update({
            "id": str(uuid.uuid4()),
            "interview_status": "no",
            "hired_status": "no interview",
            "tags": [],
            "interview_summaries": [],
            "metadata": metadata or {},
            "embedding": embed_text(text)  
        })

        if 'embedding' in parsed and isinstance(parsed['embedding'], list):
            from decimal import Decimal
            parsed['embedding'] = [Decimal(str(x)) for x in parsed['embedding']]


        return parsed

    except Exception as e:
        print(f"💥 Error extracting profile with LLM: {e}")
        return None

def process_resume(document: Document):
    # Check if we already have this resume processed
    if document.metadata.get('source'):
        existing_resume = dynamodb_handler.get_resume(document.metadata['source'])
        if existing_resume:
            return existing_resume
    
    # Process new resume
    extracted_data = extract_profile_using_llm(document.page_content, metadata=document.metadata)
    
    if extracted_data:
        # Add type field to distinguish between resumes and job descriptions
        extracted_data['type'] = 'resume'
        # Save to DynamoDB
        dynamodb_handler.save_resume(extracted_data)
    
    return extracted_data

    #return extract_profile_using_llm(document.page_content, metadata=document.metadata)

if __name__ == "__main__":
    sample_resume_text = """
    John Doe
    john.doe@email.com | (123) 456-7890 | linkedin.com/in/johndoe

    Experience:
    January 2023 - Present
    Software Engineer
    Google
    - Developed and maintained backend APIs using Python and Django.
    - Worked with PostgreSQL database.
    - Deployed applications on Google Cloud Platform.

    September 2020 - December 2022
    Junior Developer
    Amazon
    - Assisted senior developers in building web applications using Java and Spring.
    - Wrote unit and integration tests.
    - Participated in agile development meetings.

    Education:
    Master of Science in Computer Science, University of Cape Town, 2022
    B.Sc. Information Technology, Stellenbosch University, 2020

    Projects:
    Project A: Built a thing using Python.
    Technologies: Python.
    """
    sample_document = Document(page_content=sample_resume_text, metadata={"source": "sample.txt"})
    extracted_info = process_resume(sample_document)

    print("\nExtracted Information from Sample:")
    print(json.dumps(extracted_info, indent=4))
