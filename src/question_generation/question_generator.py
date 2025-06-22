# src/question_generation/question_generator.py
import boto3
import json

bedrock = boto3.client(service_name='bedrock-runtime', region_name='eu-central-1') # Ensure your region is correct

def generate_interview_questions(resume_data, job_description_data, num_questions=10):
    """Generates interview questions using AWS Bedrock's Messages API."""
    model_id = 'anthropic.claude-3-haiku-20240307-v1:0'  

    messages = [
        {
            "role": "user",
            "content": f"""You are an expert interview question generator. Based on the following resume data:
            {resume_data}
            and the following job description:
            {job_description_data}
            generate {num_questions} relevant interview questions that would help assess the candidate's suitability for the role. Focus on skills, experience, and alignment with the job requirements. Format each question on a new line."""
        }
    ]

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9
    })

    accept = 'application/json'
    content_type = 'application/json'

    try:
        response = bedrock.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)
        response_body = json.loads(response.get('body').read())
        generated_text = response_body['content'][0]['text'] # Adjust response parsing for Messages API
        return generated_text.split("\n")
    except Exception as e:
        print(f"Error during Bedrock Messages API invocation: {e}")
        return [f"Error generating questions (Messages API): {e}"]

if __name__ == "__main__":
    sample_resume_data = {
        "skills": ["Python", "Django"],
        "experience": [{"title": "Software Engineer", "company": "Acme"}]
    }
    sample_job_description_data = {
        "skills": ["Python", "Flask", "AWS"],
        "responsibilities": ["Develop web applications using Python and Flask."]
    }
    questions = generate_interview_questions(sample_resume_data, sample_job_description_data)
    for q in questions:
        print(q)
        print(q)
        