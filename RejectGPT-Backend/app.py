from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import json
from services.parser import ResumeParser
from services.link_checker import LinkChecker
from services.analyzer import ResumeAnalyzer

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for resume data
resume_storage = {}

# Initialize services
parser = ResumeParser()
link_checker = LinkChecker()
analyzer = ResumeAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "RejectGPT Backend is running!"})

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({
            "status": "error", 
            "message": "No file uploaded. Did you forget to attach your masterpiece?"
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            "status": "error", 
            "message": "No file selected. We're not mind readers!"
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error", 
            "message": "Only PDF and DOCX files allowed. We don't speak PowerPoint!"
        }), 400
    
    try:
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                "status": "error", 
                "message": "File too large! Your resume shouldn't be longer than a novel."
            }), 400
        
        # Generate unique ID and save file
        resume_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{resume_id}_{filename}")
        file.save(file_path)
        
        # Parse resume
        parsed_data = parser.parse_resume(file_path)
        
        # Store parsed data
        resume_storage[resume_id] = {
            'file_path': file_path,
            'filename': filename,
            'parsed_data': parsed_data
        }
        
        return jsonify({
            "status": "success", 
            "resume_id": resume_id,
            "message": "Resume uploaded successfully! Prepare for some harsh truths."
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Failed to process your resume. Maybe it's as broken as your career prospects? Error: {str(e)}"
        }), 500

@app.route('/analyze/<resume_id>', methods=['GET'])
def analyze_resume(resume_id):
    if resume_id not in resume_storage:
        return jsonify({
            "status": "error", 
            "message": "Resume not found. Did you make it up like your skills?"
        }), 404
    
    try:
        resume_data = resume_storage[resume_id]
        parsed_data = resume_data['parsed_data']
        
        # Check links
        links_analysis = link_checker.validate_links(parsed_data.get('links', []))
        
        # Generate AI analysis and rejection letter
        analysis_result = analyzer.analyze_resume(parsed_data)
        
        # Calculate score
        score_result = analyzer.calculate_score(parsed_data, links_analysis)
        
        return jsonify({
            "status": "success",
            "rejection_letter": analysis_result['rejection_letter'],
            "links": links_analysis,
            "score": score_result,
            "buzzwords_detected": analysis_result['buzzwords'],
            "issues_found": analysis_result['issues']
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Analysis failed. Your resume broke our AI. Error: {str(e)}"
        }), 500

@app.route('/excuse', methods=['GET'])
def get_excuse():
    try:
        with open('services/excuses.json', 'r') as f:
            excuses_data = json.load(f)
        
        import random
        excuse = random.choice(excuses_data['excuses'])
        
        return jsonify({
            "status": "success",
            "excuse": excuse
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Even our excuse generator gave up. Error: {str(e)}"
        }), 500

@app.route('/compare', methods=['POST'])
def compare_with_jd():
    try:
        data = request.get_json()
        resume_id = data.get('resume_id')
        job_description = data.get('job_description')
        
        if not resume_id or not job_description:
            return jsonify({
                "status": "error",
                "message": "Missing resume_id or job_description. Can't compare nothing to nothing!"
            }), 400
        
        if resume_id not in resume_storage:
            return jsonify({
                "status": "error",
                "message": "Resume not found. Stop making things up!"
            }), 404
        
        resume_data = resume_storage[resume_id]
        parsed_data = resume_data['parsed_data']
        
        # Perform comparison
        comparison_result = analyzer.compare_with_job_description(parsed_data, job_description)
        
        return jsonify({
            "status": "success",
            "mismatch": comparison_result['mismatch_commentary'],
            "similarity_score": comparison_result['similarity_score'],
            "missing_skills": comparison_result['missing_skills']
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Comparison failed spectacularly. Error: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found. Are you lost like your career?"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "message": "Internal server error. Our system is having an existential crisis."
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)