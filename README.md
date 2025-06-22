# 🤖 AI-Powered Tech Interview Assistant

This project is designed to automate the technical interview process by leveraging AI to parse resumes, generate personalized questions, evaluate candidates, and provide a searchable database of structured feedback. Built by a team of data engineering interns, the system helps HR and engineers manage hiring pipelines more efficiently.

---

## 📌 Key Features

- 🔍 **Resume Parsing** — Extracts structured data from uploaded resumes.
- 🧠 **AI-Powered Question Generation** — Uses LLMs to create customized technical and behavioral interview questions.
- 📝 **Feedback Scoring** — Structured forms for engineers and HR to rate candidates.
- 🔎 **Candidate Search** — Find candidates using structured filters (e.g., scores) and semantic skill matching via vector embeddings.
- 🛠️ **Modular Architecture** — Designed using APIs, services, and a shared database for flexibility and scalability.

---

## 🧱 Tech Stack

| Layer | Tool |
|------|------|
| API Framework | FastAPI |
| LLM | OpenAI GPT-4 / GPT-3.5 |
| Database | PostgreSQL |
| Embedding + Search | OpenAI Embeddings + FAISS |
| Containerization | Docker / Docker Compose |
| Vector Similarity | Cosine Similarity |
| Other | Pandas, LangChain, PyMuPDF, Pydantic, Uvicorn |

---

## 🚦 System Overview

### 📂 Core Components:
- **Resume Parser API** — Parses PDF resumes into structured JSON.
- **LLM Question Generator API** — Creates technical questions based on candidate and project details.
- **Feedback API** — Stores and retrieves structured feedback.
- **Search API** — Supports both SQL filtering and semantic skill matching using embeddings.

---

## 🔍 Semantic Search Using Embeddings

We use **OpenAI Embeddings** to convert candidate summaries or skill sets into dense vectors. These vectors allow us to perform semantic searches, where queries like `"AWS Athena"` can match candidates who mentioned `"cloud data querying tools"` or `"Presto/Athena"` — even if the exact word match is missing.

### 📐 Cosine Similarity

To find the best candidate matches, we compute the **cosine similarity** between the embedding of a search query and the stored candidate vectors.

> **Formula:**  
> `cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)`

- A value closer to `1.0` means **high similarity**
- A value closer to `0.0` means **low similarity**

This approach allows for **fuzzy matching** across various synonyms and phrases, outperforming traditional SQL `LIKE` queries.

---

## 🧪 Setup Instructions

# 1. Clone the repo
git clone https://github.com/yourusername/interview-ai-assistant.git
cd interview-ai-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
uvicorn app.main:app --reload

# 5. (Optional) Run Docker
docker compose up --build


## 📊 Example Workflow

```plaintext
1. HR uploads a resume → Resume API extracts structured data (experience, skills, education, etc.)

2. System matches candidate profile to company-defined job levels (Junior, Mid-Level, Senior) using extracted CV data and internal benchmarks.

3. Candidate is stored in a level-specific talent pool for future use (e.g., junior dev pool).

4. System sends candidate data + desired role to LLM → Generates personalized interview questions.

5. Engineer conducts technical interview and submits feedback scores (e.g., problem solving, communication).

6. HR conducts final feedback round, accepts/rejects candidate, and adds final summary.

7. All structured scores and AI summaries are stored in the candidate database.

8. HR can later search for suitable candidates using filters like:
   - Personality score > 4
   - Role = "Backend Developer"
   - Job Level = "Mid"
   - Skills ≈ "AWS Athena", "ETL", "S3"

9. The system performs hybrid search:
   - SQL filters for structured fields
   - Cosine similarity via vector embeddings for semantic skill match

