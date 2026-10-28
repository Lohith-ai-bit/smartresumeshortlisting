"""
Smart Resume Shortlisting - NLP Resume Processing Engine
Extracts: Name, Email, Phone, Skills, CGPA, Experience, Education, Location
Calculates: Weighted score based on job description
Uses: regex-based skill extraction + TF-IDF cosine similarity (no heavy NLP downloads needed)
"""

import re
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from docx import Document

# ============================================================
# SKILL DATABASE (comprehensive tech skills dictionary)
# ============================================================
SKILL_DB = {
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'go', 'golang',
    'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',
    'dart', 'lua', 'bash', 'shell', 'sql', 'html', 'css', 'html5', 'css3',
    
    # Web Frameworks
    'django', 'flask', 'fastapi', 'spring', 'springboot', 'spring boot', 'express',
    'express.js', 'node.js', 'nodejs', 'react', 'react.js', 'angular', 'angular.js',
    'vue', 'vue.js', 'next.js', 'nextjs', 'nuxt.js', 'laravel', 'rails', 'ruby on rails',
    'asp.net', 'asp.net core', 'svelte', 'jquery', 'bootstrap', 'tailwind',
    
    # Data Science / ML
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
    'scikit-learn', 'sklearn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
    'nltk', 'spacy', 'hugging face', 'transformers', 'natural language processing',
    'nlp', 'computer vision', 'opencv', 'data analysis', 'data science',
    'data visualization', 'statistics', 'big data', 'hadoop', 'spark', 'pyspark',
    'hive', 'kafka', 'airflow', 'mlops', 'data mining', 'predictive modeling',
    
    # Databases
    'mysql', 'postgresql', 'postgres', 'mongodb', 'sqlite', 'oracle', 'redis',
    'elasticsearch', 'cassandra', 'dynamodb', 'firebase', 'mariadb', 'mssql',
    'neo4j', 'graphql',
    
    # Cloud & DevOps
    'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 'google cloud',
    'docker', 'kubernetes', 'k8s', 'jenkins', 'ci/cd', 'terraform', 'ansible',
    'devops', 'linux', 'git', 'github', 'gitlab', 'bitbucket', 'nginx', 'apache',
    
    # AI / NLP specific
    'natural language processing', 'nlp', 'named entity recognition', 'ner',
    'sentiment analysis', 'text classification', 'topic modeling', 'word embeddings',
    'word2vec', 'glove', 'bert', 'gpt', 'llm', 'large language models', 'rag',
    'information retrieval', 'tf-idf', 'tfidf', 'cosine similarity',
    
    # Mobile
    'android', 'ios', 'flutter', 'react native', 'xamarin',
    
    # Soft / Management
    'agile', 'scrum', 'jira', 'project management', 'team leadership',
    'communication', 'problem solving', 'critical thinking', 'teamwork',
    
    # Others
    'rest api', 'restful api', 'graphql', 'microservices', 'object-oriented programming',
    'oop', 'test driven development', 'tdd', 'unit testing', 'selenium',
    'git', 'api', 'web development', 'backend', 'frontend', 'full stack',
    'security', 'cryptography', 'blockchain', 'iot', 'embedded systems',
    'cad', 'autocad', 'solidworks', 'catia', 'revit', 'ansys', 'matlab/simulink',
    'tableau', 'power bi', 'excel', 'microsoft excel', 'ms office',
    'mechanical engineering', 'electrical engineering', 'civil engineering',
    'computer science', 'cse', 'electronics', 'communication', 'vlsi',
}

# Skill aliases for normalization
SKILL_ALIASES = {
    'nodejs': 'node.js', 'node js': 'node.js', 'expressjs': 'express', 'express js': 'express',
    'reactjs': 'react', 'react js': 'react', 'angularjs': 'angular', 'angular js': 'angular',
    'vuejs': 'vue.js', 'vue js': 'vue.js', 'nextjs': 'next.js', 'next js': 'next.js',
    'sklearn': 'scikit-learn', 'tfidf': 'tf-idf', 'gcp': 'google cloud',
    'aws': 'amazon web services', 'azure': 'microsoft azure', 'springboot': 'spring boot',
    'spring boot': 'spring boot', 'golang': 'go', 'postgresql': 'postgres',
    'ml': 'machine learning', 'dl': 'deep learning', 'cv': 'computer vision',
    'js': 'javascript', 'ts': 'typescript', 'py': 'python',
    'mysql': 'mysql', 'mongodb': 'mongodb', 'postgres': 'postgres',
    'nlp': 'natural language processing', 'ner': 'named entity recognition',
    'ci cd': 'ci/cd', 'cicd': 'ci/cd', 'restful': 'rest api', 'rest': 'rest api',
    'oops': 'oop', 'tdd': 'test driven development',
    'machine-learning': 'machine learning', 'deep-learning': 'deep learning',
    'data-science': 'data science', 'data-analysis': 'data analysis',
}

DEGREE_PATTERNS = [
    (r'\bb\s*\.?\s*tech\b|\bbachelor[\s-]?of[\s-]?technology\b|\bbtech\b', 'B.Tech'),
    (r'\bb\s*\.?\s*e\b|\bbachelor[\s-]?of[\s-]?engineering\b|\bbe\b', 'B.E'),
    (r'\bb\s*\.?\s*sc\b|\bbachelor[\s-]?of[\s-]?science\b|\bbsc\b', 'B.Sc'),
    (r'\bb\s*\.?\s*c\s*\.?\s*a\b|\bbca\b', 'BCA'),
    (r'\bm\s*\.?\s*tech\b|\bmaster[\s-]?of[\s-]?technology\b|\bmtech\b', 'M.Tech'),
    (r'\bm\s*\.?\s*e\b|\bmaster[\s-]?of[\s-]?engineering\b|\bme\b', 'M.E'),
    (r'\bm\s*\.?\s*sc\b|\bmaster[\s-]?of[\s-]?science\b|\bmsc\b', 'M.Sc'),
    (r'\bm\s*\.?\s*c\s*\.?\s*a\b|\bmca\b', 'MCA'),
    (r'\bmba\b|\bmaster[\s-]?of[\s-]?business[\s-]?administration\b', 'MBA'),
    (r'\bph\s*\.?\s*d\b|\bdoctor[\s-]?of[\s-]?philosophy\b|\bdoctorate\b', 'Ph.D'),
    (r'\bdiploma\b|\bpolytechnic\b', 'Diploma'),
    (r'\bintermediate\b|\b12th\b|\bhigher secondary\b|\b\+2\b', '12th/Higher Secondary'),
]

# Degree ranking for scoring
DEGREE_SCORES = {
    'Ph.D': 100, 'M.Tech': 90, 'M.E': 88, 'M.Sc': 82, 'MCA': 80, 'MBA': 80,
    'B.Tech': 70, 'B.E': 68, 'B.Sc': 55, 'BCA': 50, 'Diploma': 40, '12th/Higher Secondary': 30
}


def extract_text_from_pdf(filepath):
    """Extract text from PDF using pdfplumber"""
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        text = f"[PDF Error: {str(e)}]"
    return text


def extract_text_from_docx(filepath):
    """Extract text from DOCX using python-docx"""
    text = ""
    try:
        doc = Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "
                text += "\n"
    except Exception as e:
        text = f"[DOCX Error: {str(e)}]"
    return text


def extract_text(filepath):
    """Extract text from PDF or DOCX based on extension"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext in ['.docx', '.doc']:
        return extract_text_from_docx(filepath)
    else:
        # Try as plain text
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


def normalize_skill(skill):
    """Normalize skill names using aliases"""
    s = skill.lower().strip()
    s = re.sub(r'[^\w\s.+-/#]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return SKILL_ALIASES.get(s, s)


def extract_skills(text):
    """Extract skills from text using n-gram matching against skill database"""
    text_lower = text.lower()
    text_lower = re.sub(r'[^\w\s.+-/#]', ' ', text_lower)
    text_lower = re.sub(r'\s+', ' ', text_lower)
    
    found_skills = set()
    
    # Sort skills by length (longest first) to catch multi-word skills first
    sorted_skills = sorted(SKILL_DB, key=len, reverse=True)
    
    for skill in sorted_skills:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(normalize_skill(skill))
    
    # Also check aliases
    for alias, canonical in SKILL_ALIASES.items():
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(canonical)
    
    return sorted(found_skills)


def extract_email(text):
    """Extract email addresses"""
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return emails[0] if emails else None


def extract_phone(text):
    """Extract phone numbers (Indian format compatible)"""
    patterns = [
        r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]',  # General
        r'\+91[\s-]?[6-9]\d{9}',  # Indian with +91
        r'\b[6-9]\d{9}\b',  # Indian 10-digit
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0].strip()
    return None


def extract_name(text):
    """Extract candidate name - typically first line or near contact info"""
    lines = text.strip().split('\n')
    # Look at first few non-empty lines
    for line in lines[:10]:
        line = line.strip()
        if len(line) < 5 or len(line) > 50:
            continue
        # Skip lines that look like emails, phones, urls
        if '@' in line or re.match(r'^[\d\s\-\+\(\)]+$', line) or 'http' in line:
            continue
        # Lines with 2-4 words, title case, no special chars
        words = line.split()
        if 1 <= len(words) <= 4:
            # Check if looks like a name (title case, mostly letters)
            clean = re.sub(r'[^\w\s]', '', line)
            if clean and len(clean.split()) >= 1 and all(w[0].isupper() for w in clean.split() if w):
                # Exclude common section headers
                headers = {'resume', 'curriculum', 'vitae', 'cv', 'profile', 'objective',
                          'education', 'experience', 'skills', 'projects', 'summary',
                          'contact', 'address', 'email', 'phone', 'declaration'}
                if not any(h in clean.lower() for h in headers):
                    return line.strip()
    return None


def extract_cgpa(text):
    """Extract CGPA/Percentage from text"""
    cgpa = None
    percentage = None
    
    # CGPA patterns (e.g., CGPA: 8.5, GPA 3.7/4, 8.7/10)
    cgpa_patterns = [
        r'(?:cgpa|gpa|sgpa|cgpa/10|cgpa of)[\s:]*([0-9]+\.?[0-9]*)\s*(?:/[0-9]+)?',
        r'([0-9]+\.[0-9]+)\s*/\s*10\b',
        r'\bcgpa[\s\-:–]+([0-9]+\.?[0-9]*)',
    ]
    for pattern in cgpa_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                val = float(matches[0])
                if 0 < val <= 10:
                    cgpa = val
                    break
            except:
                pass
    
    # Percentage patterns (e.g., 85%, 87.5 percent)
    pct_patterns = [
        r'([0-9]+\.?[0-9]*)\s*%',
        r'([0-9]+\.?[0-9]*)\s*percent',
        r'percentage[\s:]*([0-9]+\.?[0-9]*)',
    ]
    for pattern in pct_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            try:
                val = float(m)
                if 40 <= val <= 100:
                    percentage = val
                    break
            except:
                pass
        if percentage:
            break
    
    # Convert percentage to 10-point CGPA scale if needed
    if cgpa is None and percentage is not None:
        cgpa = round(percentage / 9.5, 2) if percentage > 0 else 0
        if cgpa > 10:
            cgpa = round(cgpa / 10, 2)
    elif cgpa is not None:
        # Check if it might be on 4-point scale
        if cgpa <= 4:
            cgpa = round(cgpa * 2.5, 2)
    
    return cgpa if cgpa else 0.0


def extract_experience(text):
    """Extract years of experience"""
    years = 0
    
    # Patterns for experience
    patterns = [
        r'(\d+\+?)\s*(?:years?|yrs?|y)\s*(?:of)?\s*experience',
        r'experience[\s:]*(\d+\+?)\s*(?:years?|yrs?)',
        r'work(?:ed)?\s*(?:for)?\s*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\s*\+\s*years?',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for m in matches:
                m_clean = m.replace('+', '')
                try:
                    val = int(m_clean)
                    if 0 <= val <= 50:
                        years = max(years, val)
                except:
                    pass
    
    # Check for internships/fresher
    if years == 0:
        if re.search(r'\bintern(?:ship)?\b', text, re.IGNORECASE):
            years = 0.3  # Internship counts as ~0.3 years
        elif re.search(r'\bfresher\b|\bentry\s*level\b|\bnew\s*graduate\b', text, re.IGNORECASE):
            years = 0
        else:
            # Try counting work entries from date ranges
            date_ranges = re.findall(
                r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\s*[-–to]+\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present|current)[a-z]*\s*\d{0,4}',
                text, re.IGNORECASE
            )
            if len(date_ranges) > 0:
                years = min(len(date_ranges) * 1.0, 15)  # Rough estimate
    
    return years


def extract_education(text):
    """Extract highest degree and field of study"""
    found_degrees = []
    
    for pattern, degree_name in DEGREE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found_degrees.append(degree_name)
    
    # Determine highest degree
    if not found_degrees:
        return {'highest_degree': 'Unknown', 'degree_score': 30, 'all_degrees': []}
    
    # Find highest
    highest = max(found_degrees, key=lambda d: DEGREE_SCORES.get(d, 0))
    
    # Extract field/specialization
    fields = []
    field_keywords = [
        'computer science', 'cse', 'information technology', 'it', 'electronics',
        'electrical', 'mechanical', 'civil', 'chemical', 'aerospace', 'biotechnology',
        'artificial intelligence', 'machine learning', 'data science', 'ai & ml',
        'ai and ml', 'communication', 'telecommunication', 'ece', 'eee',
        'software engineering', 'systems', 'networking', 'cyber security',
    ]
    for f in field_keywords:
        if re.search(r'\b' + re.escape(f) + r'\b', text, re.IGNORECASE):
            fields.append(f)
    
    return {
        'highest_degree': highest,
        'degree_score': DEGREE_SCORES.get(highest, 30),
        'all_degrees': found_degrees,
        'fields': fields
    }


def extract_location(text):
    """Extract location/address"""
    # Look for common city/state patterns
    cities = ['hyderabad', 'bangalore', 'bengaluru', 'chennai', 'mumbai', 'delhi',
              'pune', 'kolkata', 'ahmedabad', 'gurgaon', 'noida', 'coimbatore',
              'vijayawada', 'visakhapatnam', 'tirupati', 'guntur', 'nellore',
              'warangal', 'kurnool', 'kadapa', 'anantapur', 'chittoor', 'tirupathi',
              'andhra pradesh', 'telangana', 'tamil nadu', 'karnataka', 'maharashtra',
              'kerala', 'new york', 'san francisco', 'london', 'berlin']
    
    text_lower = text.lower()
    found_cities = [c.title() for c in cities if re.search(r'\b' + re.escape(c) + r'\b', text_lower)]
    
    # Also look for PIN codes (Indian)
    pin_match = re.search(r'\b\d{6}\b', text)
    pin = pin_match.group(0) if pin_match else None
    
    return ', '.join(found_cities[:3]) if found_cities else (pin or None)


def parse_resume(filepath):
    """Main function: parse a resume file and extract all information"""
    text = extract_text(filepath)
    
    if not text or len(text.strip()) < 10:
        return {
            'name': 'Unknown',
            'email': None,
            'phone': None,
            'skills': [],
            'cgpa': 0.0,
            'experience_years': 0,
            'education': {'highest_degree': 'Unknown', 'degree_score': 30, 'all_degrees': [], 'fields': []},
            'location': None,
            'raw_text': text,
            'text_length': 0,
            'error': 'Could not extract text from file'
        }
    
    name = extract_name(text)
    # Fallback to filename
    if not name:
        basename = os.path.basename(filepath)
        name = os.path.splitext(basename)[0].replace('_', ' ').replace('-', ' ').title()
    
    result = {
        'name': name,
        'email': extract_email(text),
        'phone': extract_phone(text),
        'skills': extract_skills(text),
        'cgpa': extract_cgpa(text),
        'experience_years': extract_experience(text),
        'education': extract_education(text),
        'location': extract_location(text),
        'raw_text': text,
        'text_length': len(text),
        'error': None
    }
    
    return result


def compute_tfidf_similarity(job_description, resume_texts):
    """Compute TF-IDF cosine similarity between job description and resumes"""
    documents = [job_description] + resume_texts
    
    # Custom TF-IDF parameters tuned for short documents
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        # Job description is first document
        jd_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(jd_vector, resume_vectors)[0]
        return similarities
    except Exception as e:
        # Fallback: return 0.5 for all if vectorization fails
        return np.array([0.5] * len(resume_texts))


def score_resumes(job_description, parsed_resumes, weights=None):
    """
    Score resumes based on weighted criteria.
    
    Weights (default from the paper):
    - skill_match: 60%
    - cgpa: 15%
    - experience: 15%
    - education: 10%
    """
    if weights is None:
        weights = {
            'skill_match': 0.60,
            'cgpa': 0.15,
            'experience': 0.15,
            'education': 0.10,
        }
    
    # Extract required skills from job description
    jd_skills = set(extract_skills(job_description))
    jd_text_lower = job_description.lower()
    
    # Also find any explicitly mentioned skill-like words
    jd_words = set(re.findall(r'\b[a-z][a-z.+#-]{1,20}\b', jd_text_lower))
    jd_all_skills = jd_skills | {w for w in jd_words if w in SKILL_DB}
    
    # TF-IDF similarity for overall semantic match
    resume_texts = [r['raw_text'] for r in parsed_resumes]
    tfidf_scores = compute_tfidf_similarity(job_description, resume_texts)
    
    results = []
    max_exp = max((r['experience_years'] for r in parsed_resumes), default=1)
    if max_exp == 0:
        max_exp = 1
    
    for i, resume in enumerate(parsed_resumes):
        # 1. Skill matching score (Jaccard-like + TF-IDF blend)
        resume_skills_set = set(resume['skills'])
        if jd_all_skills:
            matched_skills = resume_skills_set & jd_all_skills
            skill_coverage = len(matched_skills) / len(jd_all_skills) if jd_all_skills else 0
            # Blend with TF-IDF
            skill_score = (skill_coverage * 0.7 + tfidf_scores[i] * 0.3) * 100
        else:
            skill_score = tfidf_scores[i] * 100
            matched_skills = set()
        
        # 2. CGPA score (out of 10 -> 0-100)
        cgpa_score = min((resume['cgpa'] / 10.0) * 100, 100) if resume['cgpa'] > 0 else 50
        
        # 3. Experience score (normalize to 0-100, cap at 8 years)
        exp_cap = 8.0
        exp_normalized = min(resume['experience_years'] / exp_cap, 1.0) * 100
        if resume['experience_years'] >= 3:
            exp_score = exp_normalized  # Experienced
        elif resume['experience_years'] >= 1:
            exp_score = max(exp_normalized, 40)  # 1+ year gets minimum 40
        elif resume['experience_years'] > 0:
            exp_score = 35  # Internship/fresher with internship gets 35
        else:
            exp_score = 25  # True fresher gets baseline 25
        
        # 4. Education score
        edu_score = resume['education']['degree_score']
        
        # Weighted total
        total_score = (
            weights['skill_match'] * skill_score +
            weights['cgpa'] * cgpa_score +
            weights['experience'] * exp_score +
            weights['education'] * edu_score
        )
        
        results.append({
            'resume': resume,
            'total_score': round(total_score, 2),
            'skill_score': round(skill_score, 2),
            'cgpa_score': round(cgpa_score, 2),
            'experience_score': round(exp_score, 2),
            'education_score': round(edu_score, 2),
            'tfidf_score': round(float(tfidf_scores[i]) * 100, 2),
            'matched_skills': sorted(matched_skills),
            'skill_coverage': round(len(matched_skills) / len(jd_all_skills) * 100, 1) if jd_all_skills else 0,
            'jd_skills_count': len(jd_all_skills),
        })
    
    # Sort by total score descending
    results.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Add rank
    for rank, r in enumerate(results, 1):
        r['rank'] = rank
    
    return results, jd_all_skills


if __name__ == '__main__':
    # Quick test
    print("=== Resume Processor Module Loaded ===")
    print(f"Skill database contains {len(SKILL_DB)} skills")
