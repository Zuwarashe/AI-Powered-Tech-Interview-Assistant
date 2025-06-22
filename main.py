import streamlit as st
from src.data_automation.pipelines.data_loader import load_and_extract_text_from_all_folders_s3
from src.resume_processing.information_extraction import process_resume
from src.job_description_processing.job_description_processor import extract_job_details_llm
from src.question_generation.question_generator import generate_interview_questions
from src.matching.resume_job_matcher import perform_matching
from src.matching.semantic_matcher import find_top_matches
from src.data_automation.pipelines.dynamodb_operations import DynamoDBHandler

# Initialize DynamoDB handler
dynamodb_handler = DynamoDBHandler(table_name="ResumeJobMatcher")

# Load and process data with caching
@st.cache_data(show_spinner="Loading and processing resumes and job descriptions...")

# utils.py or just above your main() function
def sanitize_resume(resume: dict, index: int = 0) -> dict:
    """
    Ensure resume dict has all required keys with default values.
    Adds fallback values to avoid runtime errors in show_resume().
    """
    return {
        "name": resume.get("name", f"Candidate {index + 1}"),
        "contact": {
            "email": resume.get("contact", {}).get("email", "Not provided"),
            "phone": resume.get("contact", {}).get("phone", "Not provided"),
            "linkedin": resume.get("contact", {}).get("linkedin", "Not provided")
        },
        "skills": resume.get("skills", []),
        "experience": resume.get("experience", []),
        "education": resume.get("education", []),
        "embedding": resume.get("embedding", [])
    }


def load_data(force_refresh: bool = False):
    bucket = "zmakarimayi-testing-data-upload"
    prefix = "Data"
    
    # First try to load from DynamoDB unless force_refresh is True
    if not force_refresh:
        resumes = dynamodb_handler.get_all_resumes()
        jds = dynamodb_handler.get_all_job_descriptions()
        
        if resumes and jds:
            return resumes, jds
    
    # If force_refresh or no data in DynamoDB, load from S3
    raw_data = load_and_extract_text_from_all_folders_s3(bucket, prefix)
    
    # Process resumes
    resumes = []
    for doc in raw_data.get("Resumes", []):
        processed_resume = process_resume(doc)
        if processed_resume:
            resumes.append(processed_resume)
    
    # Process job descriptions
    jds = []
    for doc in raw_data.get("job_descriptions", []):
        if doc:
            extracted_levels = extract_job_details_llm(doc)
            if extracted_levels:
                jds.extend(extracted_levels)
    
    return resumes, jds


def show_resume(resume):
    st.subheader("👤 Candidate Details")

    name = resume.get('name', 'Unnamed Candidate') if isinstance(resume, dict) else 'Unnamed Candidate'
    st.markdown(f"**Name:** {name}")

    contact = resume.get('contact', {}) if isinstance(resume, dict) else {}
    if isinstance(contact, dict):
        if contact.get('email'):
            st.markdown(f"**Email:** {contact['email']}")
        if contact.get('phone'):
            st.markdown(f"**Phone:** {contact['phone']}")
        if contact.get('linkedin'):
            st.markdown(f"**LinkedIn:** {contact['linkedin']}")

    if resume.get("education"):
        st.subheader("🎓 Education")
        for edu in resume["education"]:
            if isinstance(edu, dict):
                st.markdown(
                    f"- **{edu.get('degree', 'Degree')}**, "
                    f"{edu.get('major', 'N/A')} at {edu.get('institution', 'Institution')} "
                    f"({edu.get('year', 'N/A')})"
                )
            else:
                st.markdown(f"- {edu}")

    if resume.get("experience"):
        st.subheader("💼 Experience")
        for exp in resume["experience"]:
            if isinstance(exp, dict):
                st.markdown(
                    f"- **{exp.get('title', 'Role')}**, "
                    f"{exp.get('company', 'Company')} ({exp.get('duration', 'N/A')})"
                )
                for resp in exp.get("responsibilities", []):
                    st.markdown(f"  • {resp}")
            else:
                st.markdown(f"- {exp}")

    if resume.get("skills"):
        st.subheader("🛠️ Skills")
        if isinstance(resume["skills"], list):
            st.markdown(", ".join(resume["skills"]))
        elif isinstance(resume["skills"], str):
            st.markdown(resume["skills"])



def show_job_description(jd):
    st.subheader("📋 Job Description")
    st.markdown(f"**Level:** {jd.get('level', 'N/A')}")
    st.markdown(f"**Title:** {jd.get('title', 'N/A')}")
    st.markdown(f"**Experience:** {jd.get('experience', 'N/A')}")
    st.markdown(f"**Focus:** {jd.get('focus', 'N/A')}")

    if jd.get("core_requirements"):
        st.markdown("**Core Requirements:**")
        for item in jd["core_requirements"]:
            st.markdown(f"- {item}")

    if jd.get("soft_skills"):
        st.markdown("**Soft Skills:**")
        for skill in jd["soft_skills"]:
            st.markdown(f"- {skill}")

    if jd.get("technologies_mentioned"):
        st.markdown("**Technologies Mentioned:**")
        st.markdown(", ".join(jd["technologies_mentioned"]))

    # if jd.get("full_text"):
    #     with st.expander("See original text"):
    #         st.text(jd["full_text"])

def show_semantic_matches(jd, resumes, threshold=0.3):
    st.subheader(f"🔍 Top Matches (Threshold: {threshold:.0%})")
    
    with st.spinner(f"Scanning {len(resumes)} resumes..."):
        # In your main code before calling show_semantic_matches:
        # print("Before semantic matching:")
        print(f"JD keys: {jd}")
        print(f"First resume keys: {resumes[0].keys() if resumes else 'No resumes'}")
        print(f"First resume content sample: {dict(list(resumes[0].items())[:3]) if resumes else 'No resumes'}")
        matches = find_top_matches(jd, resumes, similarity_threshold=threshold)
        #st.write("Debug - First match data:", matches[0] if matches else "No matches")
        #st.write("Debug - First match data:", matches[1] if matches else "No matches")
        
    if not matches:
        st.warning("No matches meeting the current threshold. Try lowering the similarity requirement.")
        return
    
    # # Visual score distribution
    # scores = [m['score'] for m in matches if isinstance(m.get('score'), (int, float))]
    # if scores:
    #     st.vega_lite_chart({
    #         "mark": {"type": "bar", "cornerRadiusEnd": 4},
    #         "encoding": {
    #             "x": {"field": "score", "bin": {"maxbins": 20}, "title": "Similarity Score"},
    #             "y": {"aggregate": "count", "title": "Number of Candidates"}
    #         }
    #     }, use_container_width=True)

    # Top matches display
    for i, match in enumerate(matches[:10], 1):
        resume = match.get('resume')
        if not isinstance(resume, dict):
            st.warning(f"Skipping match #{i}: Resume data is invalid or missing.")
            continue

        resume_name = resume.get('name', f"Candidate {i}")
        with st.expander(f"🏅 #{i}: {resume_name} (Score: {match.get('score', 0):.0%})", expanded=i==1):
            show_resume(resume)
            
            # # Action buttons
            # cols = st.columns(3)
            # with cols[0]:
            #     if st.button("📊 Full Analysis", key=f"full_{i}"):
            #         st.session_state['detailed_match'] = match
            # with cols[1]:
            #     if st.button("❓ Questions", key=f"questions_{i}"):
            #         questions = generate_interview_questions(
            #             resume, jd, num_questions=5
            #         )
            #         st.session_state['questions'] = questions
            # with cols[2]:
            #     if st.button("📋 Compare", key=f"compare_{i}"):
            #         st.session_state['compare_resume'] = resume

    # Show generated questions if available
    if 'questions' in st.session_state:
        st.divider()
        st.subheader("Generated Questions")
        for q in st.session_state['questions']:
            st.markdown(f"- {q}")



def show_matching_results(results):
    st.subheader("🎯 Match Results")
    score = results.get("score", 0.0)
    st.markdown(f"**Match Score:** {score * 100:.1f}%")

    if "skills_matching" in results:
        st.markdown("**Skills Matching:**")
        for skill, matched in results["skills_matching"].items():
            st.markdown(f"- {skill}: {'✅' if matched else '❌'}")

    if "experience_matching" in results:
        st.markdown("**Experience Matching:**")
        for exp, matched in results["experience_matching"].items():
            st.markdown(f"- {exp}: {'✅' if matched else '❌'}")

def main():
    st.title("🧠 Resume & Job Matcher")
    
    # Add refresh button in sidebar
    st.sidebar.title("Options")
    force_refresh = st.sidebar.checkbox("Force refresh from S3", value=False)
    
    # Load data
    with st.spinner("Loading data..."):
        resumes, jds = load_data(force_refresh=force_refresh)

    if not resumes or not jds:
        st.error("Could not load resumes or job descriptions.")
        if st.button("Retry"):
            st.rerun()
        return

    # Selection interface
    st.sidebar.title("Select Resume & Job")
    
    # Group job descriptions by level for better organization
    jd_levels = {}
    for jd in jds:
        level = jd.get('level', 'Other')
        if level not in jd_levels:
            jd_levels[level] = []
        jd_levels[level].append(jd)
    
    # Resume selection
    resume_names = [r.get('name', f"Candidate {i+1}") for i, r in enumerate(resumes)]
    selected_resume_idx = st.sidebar.selectbox(
        "📄 Choose a Resume", 
        range(len(resumes)), 
        format_func=lambda i: resume_names[i]
    )
    
    # Job description selection with level grouping
    selected_level = st.sidebar.selectbox(
        "📌 Select Job Level",
        sorted(jd_levels.keys())
    )
    
    level_jds = jd_levels[selected_level]
    jd_titles = [
        f"{jd.get('title', f'Job {i+1}')} ({jd.get('experience', 'N/A')})" 
        for i, jd in enumerate(level_jds)
    ]
    
    selected_jd_idx = st.sidebar.selectbox(
        "📑 Choose a Job Description", 
        range(len(level_jds)), 
        format_func=lambda i: jd_titles[i]
    )

    # st.sidebar.slider(
    #     "Similarity threshold",
    #     min_value=0.0,
    #     max_value=1.0,
    #     value=0.7,
    #     step=0.05,
    #     key="similarity_threshold"
    # )

    # Replace the slider with:
    similarity_threshold = st.sidebar.slider(
        "Semantic similarity threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.3,
        step=0.01,
        format="%.2f",
        help="Higher values = stricter matching"
    )

    selected_jd = level_jds[selected_jd_idx]
    resumes = [sanitize_resume(r, idx) for idx, r in enumerate(resumes)]
    # Then pass to show_semantic_matches:
    show_semantic_matches(selected_jd, resumes, threshold=similarity_threshold)

    # Display selected resume and JD
    resume = resumes[selected_resume_idx]
    jd = level_jds[selected_jd_idx]

    # col1, col2 = st.columns(2)
    # with col1:
    #     show_resume(resume)
    # with col2:
    #     show_job_description(jd)

    # Matching and question generation
    tab1, tab2 = st.tabs(["Matching Analysis", "Interview Questions"])
    
    with tab1:
        if st.button("⚖️ Run Matching Analysis", use_container_width=True):
            with st.spinner("Analyzing match..."):
                match_results = perform_matching(resume, jd)
                show_matching_results(match_results)
    
    with tab2:
        if st.button("❓ Generate Interview Questions", use_container_width=True):
            with st.spinner("Generating questions..."):
                questions = generate_interview_questions(resume, jd, num_questions=10)
                st.subheader("🧠 Suggested Interview Questions")
                for i, q in enumerate(questions, 1):
                    st.markdown(f"{i}. {q}")

    if st.sidebar.checkbox("🎯 Show semantic matching"):
        selected_jd = level_jds[selected_jd_idx]
        resumes = [sanitize_resume(r, idx) for idx, r in enumerate(resumes)]
        show_semantic_matches(selected_jd, resumes)

if __name__ == "__main__":
    main()