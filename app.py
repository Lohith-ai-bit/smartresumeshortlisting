"""
Smart Resume Shortlisting System - Flask Web Application
End-to-end web app for automated resume screening using NLP
"""

import os
import json
import uuid
import sqlite3
from datetime import datetime
from flask import (
    Flask, request, render_template, redirect, url_for,
    flash, jsonify, send_from_directory, session
)
from werkzeug.utils import secure_filename

from resume_processor import parse_resume, score_resumes, SKILL_DB

# ============================================================
# Flask App Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DB_PATH = os.path.join(BASE_DIR, 'resume_app.db')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart-resume-shortlisting-2024-secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# Database Initialization
# ============================================================
def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema"""
    conn = get_db()
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        required_skills TEXT,
        min_cgpa REAL DEFAULT 0,
        min_experience INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER,
        candidate_name TEXT,
        email TEXT,
        phone TEXT,
        location TEXT,
        cgpa REAL,
        experience_years REAL,
        highest_degree TEXT,
        skills TEXT,
        total_score REAL,
        skill_score REAL,
        cgpa_score REAL,
        experience_score REAL,
        education_score REAL,
        tfidf_score REAL,
        matched_skills TEXT,
        skill_coverage REAL,
        rank INTEGER,
        filename TEXT,
        filepath TEXT,
        raw_text TEXT,
        status TEXT DEFAULT 'pending',
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES jobs (id)
    )''')
    
    conn.commit()
    conn.close()


# ============================================================
# Helper Functions
# ============================================================
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_job(job_id):
    """Get a single job by ID"""
    conn = get_db()
    job = conn.execute('SELECT * FROM jobs WHERE id = ?', (job_id,)).fetchone()
    conn.close()
    return dict(job) if job else None


def get_jobs():
    """Get all jobs"""
    conn = get_db()
    jobs = conn.execute('SELECT * FROM jobs ORDER BY created_at DESC').fetchall()
    conn.close()
    return [dict(j) for j in jobs]


def get_resumes_for_job(job_id):
    """Get all resumes for a job, ordered by rank"""
    conn = get_db()
    resumes = conn.execute(
        'SELECT * FROM resumes WHERE job_id = ? ORDER BY rank ASC, total_score DESC',
        (job_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in resumes]


def delete_resume(resume_id):
    """Delete a resume record and its file"""
    conn = get_db()
    resume = conn.execute('SELECT * FROM resumes WHERE id = ?', (resume_id,)).fetchone()
    if resume:
        # Delete file
        try:
            if resume['filepath'] and os.path.exists(resume['filepath']):
                os.remove(resume['filepath'])
        except:
            pass
        conn.execute('DELETE FROM resumes WHERE id = ?', (resume_id,))
        conn.commit()
    conn.close()


def delete_job(job_id):
    """Delete a job and all its resumes"""
    conn = get_db()
    resumes = conn.execute('SELECT * FROM resumes WHERE job_id = ?', (job_id,)).fetchall()
    for r in resumes:
        try:
            if r['filepath'] and os.path.exists(r['filepath']):
                os.remove(r['filepath'])
        except:
            pass
    conn.execute('DELETE FROM resumes WHERE job_id = ?', (job_id,))
    conn.execute('DELETE FROM jobs WHERE id = ?', (job_id,))
    conn.commit()
    conn.close()


def process_and_score_resume(job_id, filepath, filename):
    """Parse a resume, score it against the job, and save to DB"""
    job = get_job(job_id)
    if not job:
        return None, "Job not found"
    
    # Parse resume
    parsed = parse_resume(filepath)
    if parsed.get('error') and parsed['text_length'] < 20:
        return None, f"Failed to parse resume: {parsed.get('error', 'Unknown error')}"
    
    # Score against job description
    from resume_processor import score_resumes
    results, jd_skills = score_resumes(job['description'], [parsed])
    result = results[0]
    
    # Save to database
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO resumes (
        job_id, candidate_name, email, phone, location, cgpa, experience_years,
        highest_degree, skills, total_score, skill_score, cgpa_score,
        experience_score, education_score, tfidf_score, matched_skills,
        skill_coverage, rank, filename, filepath, raw_text, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
    (
        job_id,
        result['resume']['name'],
        result['resume']['email'],
        result['resume']['phone'],
        result['resume']['location'],
        result['resume']['cgpa'],
        result['resume']['experience_years'],
        result['resume']['education']['highest_degree'],
        json.dumps(result['resume']['skills']),
        result['total_score'],
        result['skill_score'],
        result['cgpa_score'],
        result['experience_score'],
        result['education_score'],
        result['tfidf_score'],
        json.dumps(result['matched_skills']),
        result['skill_coverage'],
        0,  # rank will be updated after re-ranking
        filename,
        filepath,
        result['resume']['raw_text'][:5000],  # Limit raw text storage
        'scored'
    ))
    resume_id = c.lastrowid
    conn.commit()
    conn.close()
    
    # Re-rank all resumes for this job
    re_rank_resumes(job_id)
    
    return resume_id, None


def re_rank_resumes(job_id):
    """Re-rank all resumes for a job based on their scores"""
    conn = get_db()
    resumes = conn.execute(
        'SELECT id, total_score FROM resumes WHERE job_id = ? ORDER BY total_score DESC',
        (job_id,)
    ).fetchall()
    for rank, r in enumerate(resumes, 1):
        conn.execute('UPDATE resumes SET rank = ? WHERE id = ?', (rank, r['id']))
    conn.commit()
    conn.close()


def batch_score_all(job_id):
    """Re-score all resumes for a job (e.g., after job description changes)"""
    conn = get_db()
    job = conn.execute('SELECT * FROM jobs WHERE id = ?', (job_id,)).fetchone()
    resumes = conn.execute('SELECT * FROM resumes WHERE job_id = ?', (job_id,)).fetchall()
    conn.close()
    
    if not job or not resumes:
        return
    
    parsed_list = []
    resume_ids = []
    for r in resumes:
        filepath = r['filepath']
        if filepath and os.path.exists(filepath):
            parsed = parse_resume(filepath)
        else:
            # Reconstruct from stored data
            parsed = {
                'name': r['candidate_name'],
                'email': r['email'],
                'phone': r['phone'],
                'skills': json.loads(r['skills']) if r['skills'] else [],
                'cgpa': r['cgpa'] or 0,
                'experience_years': r['experience_years'] or 0,
                'education': {'highest_degree': r['highest_degree'] or 'Unknown',
                              'degree_score': 70, 'all_degrees': [], 'fields': []},
                'location': r['location'],
                'raw_text': r['raw_text'] or '',
                'text_length': len(r['raw_text'] or ''),
            }
        parsed_list.append(parsed)
        resume_ids.append(r['id'])
    
    if parsed_list:
        results, _ = score_resumes(dict(job)['description'], parsed_list)
        
        conn = get_db()
        for i, result in enumerate(results):
            rid = resume_ids[i]
            conn.execute('''UPDATE resumes SET
                total_score = ?, skill_score = ?, cgpa_score = ?,
                experience_score = ?, education_score = ?, tfidf_score = ?,
                matched_skills = ?, skill_coverage = ?, rank = ?
                WHERE id = ?''',
            (
                result['total_score'], result['skill_score'], result['cgpa_score'],
                result['experience_score'], result['education_score'], result['tfidf_score'],
                json.dumps(result['matched_skills']), result['skill_coverage'],
                result['rank'], rid
            ))
        conn.commit()
        conn.close()


# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    """Home page - dashboard with job list"""
    jobs = get_jobs()
    # Add resume count to each job
    conn = get_db()
    for job in jobs:
        count = conn.execute(
            'SELECT COUNT(*) as cnt FROM resumes WHERE job_id = ?', (job['id'],)
        ).fetchone()['cnt']
        job['resume_count'] = count
        top = conn.execute(
            'SELECT candidate_name, total_score FROM resumes WHERE job_id = ? ORDER BY total_score DESC LIMIT 1',
            (job['id'],)
        ).fetchone()
        job['top_candidate'] = dict(top) if top else None
    conn.close()
    return render_template('index.html', jobs=jobs)


@app.route('/job/create', methods=['GET', 'POST'])
def create_job():
    """Create a new job posting"""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        description = request.form.get('description', '').strip()
        min_cgpa = float(request.form.get('min_cgpa', 0) or 0)
        min_experience = int(request.form.get('min_experience', 0) or 0)
        
        if not title or not description:
            flash('Job title and description are required', 'error')
            return render_template('create_job.html')
        
        # Extract required skills from description
        from resume_processor import extract_skills
        required_skills = extract_skills(description)
        
        conn = get_db()
        c = conn.cursor()
        c.execute('''INSERT INTO jobs (title, description, required_skills, min_cgpa, min_experience)
                     VALUES (?, ?, ?, ?, ?)''',
                  (title, description, json.dumps(required_skills), min_cgpa, min_experience))
        job_id = c.lastrowid
        conn.commit()
        conn.close()
        
        flash(f'Job "{title}" created successfully! Now upload resumes.', 'success')
        return redirect(url_for('job_detail', job_id=job_id))
    
    return render_template('create_job.html')


@app.route('/job/<int:job_id>')
def job_detail(job_id):
    """Job detail page with results"""
    job = get_job(job_id)
    if not job:
        flash('Job not found', 'error')
        return redirect(url_for('index'))
    
    resumes = get_resumes_for_job(job_id)
    for r in resumes:
        r['skills_list'] = json.loads(r['skills']) if r['skills'] else []
        r['matched_list'] = json.loads(r['matched_skills']) if r['matched_skills'] else []
    
    job['required_skills_list'] = json.loads(job['required_skills']) if job['required_skills'] else []
    
    # Statistics
    total = len(resumes)
    shortlisted = [r for r in resumes if r['total_score'] >= 60]
    if resumes:
        avg_score = round(sum(r['total_score'] for r in resumes) / total, 1) if total > 0 else 0
        max_score = resumes[0]['total_score'] if resumes else 0
    else:
        avg_score = 0
        max_score = 0
    
    stats = {
        'total': total,
        'shortlisted': len(shortlisted),
        'rejected': total - len(shortlisted),
        'avg_score': avg_score,
        'max_score': max_score,
    }
    
    return render_template('job_detail.html', job=job, resumes=resumes, stats=stats)


@app.route('/job/<int:job_id>/upload', methods=['POST'])
def upload_resume(job_id):
    """Upload one or more resumes for a job"""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if 'resumes' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('job_detail', job_id=job_id))
    
    files = request.files.getlist('resumes')
    uploaded_count = 0
    error_count = 0
    errors = []
    
    for file in files:
        if file.filename == '':
            continue
        if file and allowed_file(file.filename):
            # Generate unique filename
            original = secure_filename(file.filename)
            ext = os.path.splitext(original)[1].lower()
            unique_name = f"{uuid.uuid4().hex}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(filepath)
            
            resume_id, error = process_and_score_resume(job_id, filepath, original)
            if resume_id:
                uploaded_count += 1
            else:
                error_count += 1
                errors.append(f"{original}: {error}")
                # Clean up failed file
                try:
                    os.remove(filepath)
                except:
                    pass
        else:
            error_count += 1
            errors.append(f"{file.filename}: Unsupported file format (use PDF, DOCX, DOC, TXT)")
    
    if uploaded_count > 0:
        flash(f'Successfully processed {uploaded_count} resume(s)!', 'success')
    if errors:
        for e in errors:
            flash(e, 'error')
    
    return redirect(url_for('job_detail', job_id=job_id))


@app.route('/job/<int:job_id>/resume/<int:resume_id>/delete', methods=['POST'])
def delete_resume_route(job_id, resume_id):
    """Delete a resume"""
    delete_resume(resume_id)
    re_rank_resumes(job_id)
    flash('Resume deleted', 'success')
    return redirect(url_for('job_detail', job_id=job_id))


@app.route('/job/<int:job_id>/delete', methods=['POST'])
def delete_job_route(job_id):
    """Delete a job"""
    delete_job(job_id)
    flash('Job deleted successfully', 'success')
    return redirect(url_for('index'))


@app.route('/job/<int:job_id>/rescore', methods=['POST'])
def rescore_job(job_id):
    """Re-score all resumes for a job"""
    batch_score_all(job_id)
    re_rank_resumes(job_id)
    flash('All resumes re-scored successfully!', 'success')
    return redirect(url_for('job_detail', job_id=job_id))


@app.route('/job/<int:job_id>/update', methods=['POST'])
def update_job(job_id):
    """Update job description and re-score"""
    title = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    min_cgpa = float(request.form.get('min_cgpa', 0) or 0)
    min_experience = int(request.form.get('min_experience', 0) or 0)
    
    if not title or not description:
        flash('Title and description are required', 'error')
        return redirect(url_for('job_detail', job_id=job_id))
    
    from resume_processor import extract_skills
    required_skills = extract_skills(description)
    
    conn = get_db()
    conn.execute('''UPDATE jobs SET title=?, description=?, required_skills=?, min_cgpa=?, min_experience=?
                    WHERE id=?''',
                 (title, description, json.dumps(required_skills), min_cgpa, min_experience, job_id))
    conn.commit()
    conn.close()
    
    # Re-score all resumes with new JD
    batch_score_all(job_id)
    re_rank_resumes(job_id)
    
    flash('Job updated and resumes re-scored!', 'success')
    return redirect(url_for('job_detail', job_id=job_id))


@app.route('/resume/<int:resume_id>/view')
def view_resume(resume_id):
    """View a single resume's details"""
    conn = get_db()
    resume = conn.execute('SELECT * FROM resumes WHERE id = ?', (resume_id,)).fetchone()
    conn.close()
    
    if not resume:
        flash('Resume not found', 'error')
        return redirect(url_for('index'))
    
    resume = dict(resume)
    resume['skills_list'] = json.loads(resume['skills']) if resume['skills'] else []
    resume['matched_list'] = json.loads(resume['matched_skills']) if resume['matched_skills'] else []
    
    job = get_job(resume['job_id'])
    return render_template('resume_detail.html', resume=resume, job=job)


@app.route('/api/skills')
def api_skills():
    """Return all recognized skills (for autocomplete)"""
    return jsonify(sorted(SKILL_DB))


@app.route('/api/job/<int:job_id>/results')
def api_results(job_id):
    """Return results as JSON (for charts)"""
    resumes = get_resumes_for_job(job_id)
    data = []
    for r in resumes:
        data.append({
            'rank': r['rank'],
            'name': r['candidate_name'],
            'score': r['total_score'],
            'skills': json.loads(r['skills']) if r['skills'] else [],
            'cgpa': r['cgpa'],
            'experience': r['experience_years'],
            'degree': r['highest_degree'],
            'matched': json.loads(r['matched_skills']) if r['matched_skills'] else [],
        })
    return jsonify(data)


# ============================================================
# Run App
# ============================================================
if __name__ == '__main__':
    init_db()
    print("=" * 60)
    print("  Smart Resume Shortlisting System")
    print("  Running on http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
