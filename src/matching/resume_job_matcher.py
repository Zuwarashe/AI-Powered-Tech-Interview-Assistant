# src/matching/resume_job_matcher.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def match_skills(resume_skills, job_skills):
    """
    Matches skills from a resume to the skills required in a job description.

    Args:
        resume_skills (list): A list of skills extracted from the resume (lowercase).
        job_skills (list): A list of skills required in the job description (lowercase).

    Returns:
        dict: A dictionary containing:
            - 'matched_skills': A list of skills present in both.
            - 'missing_skills': A list of job skills not found in the resume.
            - 'overlap_percentage': The percentage of job skills found in the resume.
    """
    if not job_skills:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "overlap_percentage": 0.0,
        }

    matched_skills = set(resume_skills) & set(job_skills)
    missing_skills = set(job_skills) - set(resume_skills)
    overlap_percentage = (len(matched_skills) / len(job_skills)) * 100 if job_skills else 0.0

    return {
        "matched_skills": sorted(list(matched_skills)),
        "missing_skills": sorted(list(missing_skills)),
        "overlap_percentage": round(overlap_percentage, 2),
    }

def compute_embedding_similarity(resume_embedding, jd_embedding):
    """
    Computes cosine similarity between resume and job description embeddings.
    """
    if resume_embedding is None or jd_embedding is None:
        return 0.0
    try:
        similarity = cosine_similarity(
            np.array(resume_embedding).reshape(1, -1),
            np.array(jd_embedding).reshape(1, -1)
        )[0][0]
        return round(similarity * 100, 2)  # Return as percentage
    except:
        return 0.0


def match_experience_keywords(resume_experience, job_description_text):
    """
    Checks if keywords from the job description are present in the resume experience.

    Args:
        resume_experience (list): A list of experience dictionaries from the resume.
        job_description_text (str): The full text of the job description (lowercase).

    Returns:
        list: A list of job description keywords found in the resume experience.
    """
    matched_keywords = set()
    if job_description_text:
        keywords = [word.lower() for word in job_description_text.split() if len(word) > 2 and word.isalnum()] # Simple keyword extraction
        for experience in resume_experience:
            experience_text = (experience.get("title", "") + " " +
                               experience.get("company", "") + " " +
                               " ".join(experience.get("responsibilities", ""))).lower()
            for keyword in keywords:
                if keyword in experience_text:
                    matched_keywords.add(keyword)
    return sorted(list(matched_keywords))

def perform_matching(resume_data, job_description_data):
    """
    Performs matching between extracted resume data and job description data.

    Args:
        resume_data (dict): Extracted data from a resume.
        job_description_data (dict): Extracted data from a job description.

    Returns:
        dict: A dictionary containing the matching results.
    """
    resume_skills = [skill.lower() for skill in resume_data.get("skills", [])]
    job_skills = [skill.lower() for skill in job_description_data.get("skills", [])]
    skills_match = match_skills(resume_skills, job_skills)

    experience_match = match_experience_keywords(
        resume_data.get("experience", []), job_description_data.get("full_text", "").lower()
    )

    embedding_similarity = compute_embedding_similarity(
        resume_data.get("embedding"), job_description_data.get("embedding")
    )

    return {
        "skills_match": skills_match,
        "experience_keywords_match": experience_match,
        "semantic_similarity_score": embedding_similarity  # This is the new part!
    }

if __name__ == "__main__":
    # Sample usage for testing
    sample_resume_data = {
        "skills": ["Python", "Django", "REST APIs", "PostgreSQL", "Machine Learning"],
        "experience": [
            {"title": "Software Engineer", "company": "Google", "responsibilities": ["Developed APIs", "Used Python"]},
            {"title": "ML Intern", "company": "Amazon", "responsibilities": ["Worked on machine learning models"]}
        ]
    }
    sample_job_description_data = {
        "skills": ["Python", "AWS", "RESTful APIs", "Machine Learning", "Docker"],
        "full_text": "Looking for a Software Engineer with experience in Python, AWS, and machine learning. Familiarity with RESTful APIs is a plus."
    }

    match_results = perform_matching(sample_resume_data, sample_job_description_data)
    import json
    print(json.dumps(match_results, indent=4))