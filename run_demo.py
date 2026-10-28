"""
Quick-start script: creates a demo job with sample resumes pre-loaded.
Run this first to see the system in action immediately.
"""
import os
import sys

# Ensure working directory is the app folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from app import app, init_db, get_db, process_and_score_resume
from resume_processor import extract_skills

def load_demo_data():
    init_db()
    conn = get_db()
    # Check if demo already exists
    existing = conn.execute("SELECT id FROM jobs WHERE title LIKE '%NLP Engineer%' LIMIT 1").fetchone()
    conn.close()
    
    if existing:
        print(f"Demo job already exists (id={existing['id']}). Skipping demo data load.")
        return existing['id']
    
    # Create demo job
    demo_jd = """We are looking for a Python Developer with experience in Natural Language Processing. 
The ideal candidate should have strong skills in Python, Machine Learning, NLP, TensorFlow or PyTorch, 
scikit-learn, pandas, and Flask/Django. Experience with SQL databases (PostgreSQL/MySQL), 
Git version control, and REST API development is required. B.Tech/B.E in Computer Science 
or related field with minimum 7 CGPA. Freshers with relevant internships/projects in NLP 
are welcome. Knowledge of spaCy, NLTK, TF-IDF, cosine similarity, word embeddings (Word2Vec/BERT), 
Deep Learning and data visualization is a plus."""

    import json
    required_skills = extract_skills(demo_jd)
    
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO jobs (title, description, required_skills, min_cgpa, min_experience)
                 VALUES (?, ?, ?, ?, ?)''',
              ('Python Developer / NLP Engineer (Demo)', demo_jd, json.dumps(required_skills), 7.0, 0))
    job_id = c.lastrowid
    conn.commit()
    conn.close()
    
    print(f"Created demo job (id={job_id}). Loading sample resumes...")
    
    # Process all sample resumes
    sample_dir = 'sample_resumes'
    for f in sorted(os.listdir(sample_dir)):
        if f.endswith('.docx') or f.endswith('.pdf'):
            path = os.path.join(sample_dir, f)
            import shutil, uuid
            # Copy to uploads with unique name
            ext = os.path.splitext(f)[1]
            unique_name = f"{uuid.uuid4().hex}{ext}"
            dest = os.path.join('uploads', unique_name)
            shutil.copy(path, dest)
            rid, err = process_and_score_resume(job_id, dest, f)
            if rid:
                print(f"  ✓ Processed {f} -> resume_id={rid}")
            else:
                print(f"  ✗ Failed {f}: {err}")
    
    return job_id


if __name__ == '__main__':
    print("=" * 60)
    print("  Smart Resume Shortlisting System - Demo Setup")
    print("=" * 60)
    job_id = load_demo_data()
    print()
    print(f"  Demo data ready! Job ID: {job_id}")
    print(f"  View results at: http://localhost:5000/job/{job_id}")
    print("=" * 60)
    print()
    print("Starting Flask server...")
    app.run(debug=False, host='0.0.0.0', port=5000)
