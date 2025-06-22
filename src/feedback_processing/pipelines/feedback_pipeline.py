# src/feedback_processing/pipelines/feedback_pipeline.py
import os
import boto3
from ..extraction.text_extractors import extract_text_from_feedback
from ..analysis.llm_analyzer import analyze_feedback_with_llm
from ..storage.dynamo_writer import write_feedback_analysis_to_dynamodb  # You'll need to create this

# Configure S3 and DynamoDB (replace with your actual settings)
S3_BUCKET_NAME = "your-s3-bucket-name"
FEEDBACK_FOLDER = "interview_feedback"  # Optional subfolder in your bucket
DYNAMODB_TABLE_NAME = "your-dynamodb-table-name"
s3 = boto3.client('s3', region_name='eu-central-1')  # Specify your region
dynamodb = boto3.resource('dynamodb', region_name='eu-central-1') # Specify your region

def process_feedback_files():
    """
    Orchestrates the processing of interview feedback files.
    """
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=FEEDBACK_FOLDER)
        if 'Contents' in response:
            feedback_files = [obj['Key'] for obj in response['Contents']]
            for file_key in feedback_files:
                if file_key.endswith(('.txt', '.pdf', '.docx')):
                    print(f"Processing feedback file: {file_key}")
                    try:
                        s3_object = s3.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
                        file_content = s3_object['Body'].read()

                        # Determine file type and decode content if necessary
                        feedback_text = ""
                        if file_key.endswith('.txt'):
                            feedback_text = file_content.decode('utf-8')
                        else:
                            # For PDF and DOCX, we'll save locally and then extract
                            local_file_path = f"/tmp/{os.path.basename(file_key)}" # Use /tmp in Lambda
                            with open(local_file_path, 'wb') as f:
                                f.write(file_content)
                            feedback_text = extract_text_from_feedback(local_file_path)
                            os.remove(local_file_path) # Clean up local file

                        if feedback_text:
                            analysis_results = analyze_feedback_with_llm(feedback_text)
                            # Prepare data for DynamoDB
                            item = {
                                'feedback_file': file_key,
                                'llm_analysis': analysis_results.get('llm_analysis'),
                                'sentiment_polarity': analysis_results.get('sentiment', {}).get('sentiment_polarity'),
                                'sentiment_subjectivity': analysis_results.get('sentiment', {}).get('sentiment_subjectivity'),
                                'keywords': analysis_results.get('keywords')
                                # Add other relevant data here
                            }
                            write_feedback_analysis_to_dynamodb(DYNAMODB_TABLE_NAME, item)
                            print(f"Analysis results stored for: {file_key}")
                        else:
                            print(f"Could not extract text from: {file_key}")
                    except Exception as e:
                        print(f"Error processing {file_key}: {e}")
        else:
            print(f"No feedback files found in {S3_BUCKET_NAME}/{FEEDBACK_FOLDER}")

    except Exception as e:
        print(f"Error listing S3 objects: {e}")

if __name__ == "__main__":
    process_feedback_files()