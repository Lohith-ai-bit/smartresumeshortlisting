# Smart Resume Shortlisting System

An AI-powered web application for automated resume screening and candidate ranking using Natural Language Processing (NLP), TF-IDF vectorization, and cosine similarity.

## 🚀 Overview

The **Smart Resume Shortlisting System** streamlines the recruitment process by parsing candidate resumes (in PDF and DOCX formats), matching extracted candidate information against customizable job descriptions, and ranking candidates based on a multi-factor weighted scoring model.

### Key Features
- **Automated Resume Parsing**: Extracts candidate name, email, phone number, location, CGPA/GPA, years of experience, highest degree, and skills from PDF and DOCX documents.
- **NLP & TF-IDF Scoring**: Leverages TF-IDF vectorization and cosine similarity to measure contextual alignment between resumes and job requirements.
- **Multi-Factor Weighted Ranking**:
  - Skill Match & Coverage (40%)
  - TF-IDF Cosine Semantic Similarity (25%)
  - Work Experience Match (15%)
  - CGPA / Academic Performance (10%)
  - Education / Degree Level (10%)
- **Interactive Web Interface**: Clean, responsive UI built with Flask, Bootstrap 5, and Chart.js for real-time visualization of scores and skill breakdowns.
- **Bulk Upload & Processing**: Support for single and batch resume uploads with instant ranking.
- **REST APIs**: Endpoints for recognized tech skills autocomplete and JSON results export.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, Flask
- **NLP & Machine Learning**: Scikit-learn (`TfidfVectorizer`, `cosine_similarity`), NumPy, SciPy
- **Document Processing**: `pdfplumber` (PDF parsing), `python-docx` (DOCX parsing)
- **Database**: SQLite3
- **Frontend**: HTML5, Jinja2, Vanilla CSS / Bootstrap 5, FontAwesome, Chart.js

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Lohith-ai-bit/smartresumeshortlisting.git
cd smartresumeshortlisting
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Application

### Option A: Run the Web App
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

### Option B: Quick-Start with Demo Data
To immediately test the application with pre-loaded job postings and sample resumes:
```bash
python run_demo.py
```
Then visit:
```
http://localhost:5000/job/1
```

---

## ⚡ Deploy to Vercel

This repository is pre-configured for zero-friction deployment on **Vercel** serverless Python:
1. Fork or push this repository to your GitHub account.
2. Go to [Vercel](https://vercel.com/) and click **Add New Project**.
3. Import the `smartresumeshortlisting` repository.
4. Keep the default settings and click **Deploy**.
5. Vercel automatically builds using `api/index.py` and `vercel.json`.

---

## 📁 Project Structure

```
smartresumeshortlisting/
├── api/
│   └── index.py            # Vercel serverless entrypoint
├── app.py                  # Main Flask application & routing logic
├── resume_processor.py     # NLP parsing, skill extraction & scoring engine
├── run_demo.py             # Script to populate demo job & sample resumes
├── requirements.txt        # Python package dependencies
├── vercel.json             # Vercel routing configuration
├── .gitignore              # Files and folders excluded from version control
├── static/
│   └── style.css           # Custom styling and animations
├── templates/
│   ├── base.html           # Base layout template
│   ├── index.html          # Dashboard & job listings
│   ├── create_job.html     # Create new job posting form
│   ├── job_detail.html     # Candidate ranking table & charts
│   └── resume_detail.html  # Detailed candidate score breakdown
├── sample_resumes/         # Sample PDF & DOCX resumes for testing
└── uploads/                # Directory for uploaded resume documents
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
