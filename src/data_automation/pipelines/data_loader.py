import boto3
import os
import tempfile
from io import BytesIO
from typing import Dict, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

s3 = boto3.client("s3")

def list_s3_files(bucket: str, prefix: str) -> List[str]:
    """List all files under a given S3 prefix (folder)"""
    files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith("/"):  # skip 'folders'
                files.append(key)
    return files

def download_s3_file(bucket: str, key: str) -> str:
    """Download S3 file to a temp file and return its path"""
    _, ext = os.path.splitext(key)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        s3.download_fileobj(bucket, key, tmp_file)
        return tmp_file.name

def load_document(bucket: str, key: str):
    ext = key.lower().split('.')[-1]
    local_path = download_s3_file(bucket, key)

    if ext == "pdf":
        loader = PyPDFLoader(local_path)
    elif ext in ["docx", "doc"]:
        loader = Docx2txtLoader(local_path)
    elif ext in ["xls", "xlsx"]:
        loader = UnstructuredExcelLoader(local_path)
    else:
        return []

    return loader.load()

def load_and_extract_text_from_all_folders_s3(bucket_name: str, base_prefix: str) -> Dict[str, List[Document]]:
    """
    Loads documents from an S3 bucket, preserving structure:
    - Resumes/
    - Project info/
    - Career Path/
    - Interview feedback/
    """
    all_documents = {}

    folder_map = {
        "Resumes": "Resumes/",
        "Career Path": "Project info/",
        "job_descriptions": "Career Path/",
        "interview_feedback": "Interview feedback/"
    }

    for label, prefix in folder_map.items():
        full_prefix = base_prefix.rstrip("/") + "/" + prefix
        files = list_s3_files(bucket_name, full_prefix)

        docs = []
        for file_key in files:
            try:
                #print(f"  Loading {label} file from S3: {file_key}") ------------------------------
                loaded_docs = load_document(bucket_name, file_key)
                docs.extend(loaded_docs)
            except Exception as e:
                print(f"  Error processing {file_key}: {e}")

        if docs:
            all_documents[label] = docs

    return all_documents
