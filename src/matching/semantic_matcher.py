import numpy as np
from numpy.linalg import norm
from decimal import Decimal
from typing import List, Dict

def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings"""
    # Convert Decimals to floats if needed
    if isinstance(embedding1[0], Decimal):
        embedding1 = [float(v) for v in embedding1]
    if isinstance(embedding2[0], Decimal):
        embedding2 = [float(v) for v in embedding2]
    
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))



def find_top_matches(
    job_description: Dict, resumes: List[Dict], top_n: int = 5,  similarity_threshold: float = 0.3
) -> List[Dict]:
    """
    Find top matching resumes for a job description based on embedding similarity
    
    Args:
        job_description: The JD dict with 'embedding' field
        resumes: List of resume dicts with 'embedding' fields and other details
        top_n: Number of top matches to return
        similarity_threshold: Minimum similarity score (0-1)
    
    Returns:
        List of dicts with 'resume' (full details) and 'score' sorted by score
    """
    if not resumes:
        return []

    if 'embedding' not in job_description:
        raise ValueError("Job description missing embedding")
    
    scored_resumes = []
    jd_embedding = job_description['embedding']
    
    for resume in resumes:
        if not isinstance(resume, dict) or 'embedding' not in resume:
            continue
        
        try:
            # Ensure we're working with proper embeddings
            resume_embedding = resume['embedding']
            if not resume_embedding:  # Skip if empty embedding
                continue
                
            # Calculate similarity
            score = cosine_similarity(jd_embedding, resume_embedding)
            
            if score >= similarity_threshold:
                # Create a new dictionary with ALL original resume data
                scored_resumes.append({
                    'resume': resume,  
                    'score': score
                })
                
        except Exception as e:
            print(f"Error processing resume: {e}")
            continue
    
    # Sort by score descending
    scored_resumes.sort(key=lambda x: x['score'], reverse=True)
    
    return scored_resumes[:top_n]