from dotenv import load_dotenv
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests  # Google OAuth Requests

request_obj = google_requests.Request()

from google_auth_oauthlib.flow import Flow
import os
load_dotenv()
import face_recognition
from flask import Flask, jsonify, request, render_template, redirect, url_for, session,send_file,send_from_directory,flash
from bson import ObjectId
import base64
import pymongo 
import numpy as np
from datetime import datetime
import datetime
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
import os
import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage
import re
import PyPDF2
import spacy
nlp = spacy.load("en_core_web_sm")
import re
app = Flask(__name__)

from io import BytesIO
from PIL import Image
import base64
import gridfs
from bson import ObjectId
from datetime import timedelta
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.exceptions import Error as CloudinaryError
cloudinary.config(
    cloud_name=os.getenv('CLOUD_NAME'),
    api_key=os.getenv('CLOUD_API_KEY'),
    api_secret=os.getenv('CLOUD_API_SECRET')
)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30) 
from pymongo import MongoClient
client = MongoClient(os.getenv("MONGO_URI"))
@app.before_request
def before_request():
    session.permanent = True  

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  

flow = Flow.from_client_config(
    client_config={
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [GOOGLE_REDIRECT_URI],
            "javascript_origins": ["http://127.0.0.1:5000"]
        }
    },
    scopes=["openid", "https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"],
    redirect_uri=GOOGLE_REDIRECT_URI
)

SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
RESUME_UPLOAD_FOLDER = 'static/resume_uploads/'
app.config['RESUME_UPLOAD_FOLDER'] = RESUME_UPLOAD_FOLDER

# Ensure the directory exists
os.makedirs(RESUME_UPLOAD_FOLDER, exist_ok=True)
# db = client['face_recognition']
db = client['face_recognition']
collection = db['users'] 
admin_collection=db["admins"]
job_applications_collection = db['job_applications']
fs = gridfs.GridFS(db) 
secret_key = os.urandom(24)
print(secret_key)
app.secret_key = secret_key

REGISTRATION_UPLOAD_FOLDER = 'static/registration_uploads/'
app.config['REGISTRATION_UPLOAD_FOLDER'] = REGISTRATION_UPLOAD_FOLDER

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ADMIN_UPLOAD_FOLDER = 'static/admin_uploads' 
app.config['ADMIN_UPLOAD_FOLDER'] = ADMIN_UPLOAD_FOLDER 

ADMIN_LOGIN_UPLOAD_FOLDER = 'static/admin_login_uploads' 
app.config['ADMIN_LOGIN_UPLOAD_FOLDER'] = ADMIN_LOGIN_UPLOAD_FOLDER 

ATTEND_UPLOAD_FOLDER = 'static/attendance_uploads' 
app.config['ATTEND_UPLOAD_FOLDER'] = ATTEND_UPLOAD_FOLDER 


EDIT_UPLOAD_FOLDER = 'static/edit_uploads'
app.config['EDIT_UPLOAD_FOLDER'] = EDIT_UPLOAD_FOLDER
def get_jobs_from_db():
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT j.id, j.title, j.description, j.skills, j.location, j.package, c.name, c.logo_url
        FROM job_descriptions j
        JOIN companies c ON j.company_id = c.id
    """)
    jobs = [
        {
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "skills": row[3],
            "location": row[4],  # Added location
            "package": row[5],   # Added package
            "company": row[6],
            "logo_url": row[7]
        }
        for row in cursor.fetchall()
    ]
    conn.close()
    return jobs

@app.route("/")
def index():
    jobs=get_jobs_from_db()
    return render_template("index.html",jobs=jobs)

@app.route("/get_jobs", methods=["GET"])
def get_jobs():
    jobs = get_jobs_from_db()
    return jsonify(jobs)
@app.route("/resume")
def resume():
    return render_template("resume.html")


@app.route("/admin_register", methods=['GET'])
def admin_register_page():
    return render_template("admin_register.html")
@app.route("/admin_register", methods=["POST"])
def admin_register():
    name = request.form.get("name")
    password = request.form.get("password")
    photo = request.files["photo"]

    if not name or not password or not photo:
        return jsonify({"success": False, "error": "All fields are required"}), 400

    # Upload photo to Cloudinary
    cloudinary_response = cloudinary.uploader.upload(photo)
    photo_url = cloudinary_response['secure_url']

    # Hash the password
    hashed_password = generate_password_hash(password)

    # Save admin data to the database
    admin_data = {
        "name": name,
        "password": hashed_password,
        "photo_url": photo_url,
        "role": "admin"
    }
    admin_collection.insert_one(admin_data)
    session['admin_name'] = name
    session['admin_role'] = 'admin'

    return jsonify({"success": True, "message": "Admin registration successful"})
@app.route("/admin_login", methods=["POST"])
def admin_login():
    
    name = request.form.get("name")
    password = request.form.get("password")
    print(name)
    print(password)

    if not name or not password:
        return jsonify({"success": False, "error": "Name and password are required"}), 400

    # Find the admin in the database
    admin = admin_collection.find_one({"name": name})
    if not admin:
        return jsonify({"success": False, "error": "Admin not found"}), 404

    # Verify the password
    if not check_password_hash(admin.get("password", ""), password):
        return jsonify({"success": False, "error": "Incorrect password"}), 401

    # Set session for the logged-in admin
    session['admin_name'] = admin['name']
    session['admin_role'] = 'admin'  # Add role for clarity
    return jsonify({"success": True, "name": admin['name']})

import requests
@app.route("/admin_login_face_get", methods=["GET"])
def admin_login_face_get():
    return render_template("admin_login_face.html")

@app.route("/admin_login_face", methods=["POST"])
def admin_login_face():
    name = request.form.get("name")
    password = request.form.get("password")
    photo = request.files['photo']
    
    # Validate input
    if not name or not password or not photo:
        return jsonify({"success": False, "error": "Invalid data"}), 400
    
    # Save the login photo to the static/uploads directory
    filename = secure_filename(photo.filename)
    login_photo_path = os.path.join(app.config['ADMIN_LOGIN_UPLOAD_FOLDER'], filename)
    photo.save(login_photo_path)

    # Load and encode the login photo
    login_image = face_recognition.load_image_file(login_photo_path)
    login_face_encodings = face_recognition.face_encodings(login_image)
    if len(login_face_encodings) == 0:
        return jsonify({"success": False, "error": "No face found in the login image"})
    
    login_face_encoding = login_face_encodings[0]

    # Iterate through all admins in the database
    for admin in admin_collection.find({"role": "admin"}):
        # Fetch the admin's photo URL from Cloudinary
        photo_url = admin['photo_url']
        
        # Download the image from Cloudinary
        response = requests.get(photo_url)
        if response.status_code != 200:
            return jsonify({"success": False, "error": "Failed to fetch admin photo from Cloudinary"})
        
        # Load the image into memory
        registered_image = Image.open(BytesIO(response.content))
        registered_image = np.array(registered_image)  # Convert to numpy array for face_recognition

        # Encode the registered admin's face
        registered_face_encodings = face_recognition.face_encodings(registered_image)
        if len(registered_face_encodings) == 0:
            continue  # Skip if no face is found in the registered image

        registered_face_encoding = registered_face_encodings[0]

        # Compare faces
        matches = face_recognition.compare_faces([registered_face_encoding], login_face_encoding)
        print("Face matches:", matches)
        print("Password matches:", check_password_hash(admin['password'], password))

        if any(matches):
            # Check if the password matches using check_password_hash
            if check_password_hash(admin['password'], password):
                session['admin_name'] = admin['name']
                session['admin_role'] = 'admin' 
                return jsonify({"success": True, "name": admin['name'], 'image_url': login_photo_path})
            else:
                return jsonify({"success": False, "error": "Incorrect password"})

    return jsonify({"success": False, "error": "No matching admin found"})
@app.route("/admin_login", methods=['GET'])
def admin_login_page():
    return render_template("admin_login.html")
@app.route("/google_login")
def google_login():
    # Redirect to Google's OAuth consent screen
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true"
    )
    session["state"] = state  # Store the state for CSRF protection
    return redirect(authorization_url)
@app.route("/success")
def success():
    # Handle the OAuth callback
    try:
        # Fetch the token from the authorization response
        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials

        # Create an instance of the Request class
        

        # Verify the OAuth token
        id_info = id_token.verify_oauth2_token(
            credentials.id_token,
            request_obj,  # Pass the instance, not the class
            GOOGLE_CLIENT_ID
        )

        # Store user information in the session
        session["email"] = id_info.get("email")
        print(f"User email: {id_info.get('email')}")

        # Check if the user exists in the database
        user = collection.find_one({"email": id_info.get("email")})
        if user:
            # If the user exists, set session variables
            session["user_name"] = user["user_name"]
            session["user_role"] = "user"
            return redirect(url_for("user_profile"))
        else:
            # If the user does not exist, redirect to the registration page
            flash("Please complete your registration.", "info")
            return redirect(url_for("user_register"))

    except Exception as e:
        print(f"Error during OAuth callback: {e}")
        return "Authentication failed. Please try again."
@app.route("/admin")
def admin():
    # Check if the session is for an admin
    if 'admin_name' not in session or session.get('admin_role') != 'admin':
        return redirect(url_for('admin_login_page'))

    # Fetch all users from the database
    users = list(collection.find({}, {"_id": 1, "name": 1, "rollno": 1, "branch": 1, 
                                      "registration_no": 1, "bio": 1, "photo_url": 1, 
                                      "ats_score": 1, "missing_skills": 1, "recommended_jobs": 1}))
    
    return render_template("admin.html", user=users)




@app.route("/user_register", methods=['GET'])
def user_register():
    return render_template('register.html')

@app.route("/register", methods=["POST"])
def register():
    try:
        # Debug: Log incoming form data
        print("Incoming form data:", request.form)
        print("Incoming files:", request.files)

        # Extract form data
        user_name = request.form.get("user_name")
        name = request.form.get("name")
        rollno = request.form.get("rollno")
        registration_no = request.form.get("registrationno")
        branch = request.form.get("branch")
        bio = request.form.get("bio")
        email = request.form.get('email')
        photo = request.files.get('photo')  # Use .get() to avoid KeyError if 'photo' is missing
        password = request.form.get("password")

        # Validate required fields
        if not user_name or not name or not rollno or not registration_no or not branch or not bio or not photo or not password:
            return jsonify({"success": False, "error": "All fields are required"}), 400

        # Debug: Log photo file details
        print("Photo file:", photo.filename, photo.content_type)

        # Upload photo to Cloudinary
        try:
            cloudinary_response = cloudinary.uploader.upload(photo)
            photo_url = cloudinary_response['secure_url']
            print("Cloudinary response:", cloudinary_response)
        except Exception as e:
            print("Cloudinary upload failed:", str(e))
            return jsonify({"success": False, "error": "Failed to upload photo to Cloudinary"}), 500

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Create user data
        user_data = {
            "user_name": user_name,
            "name": name,
            "rollno": int(rollno),
            "registration_no": int(registration_no),
            "branch": branch,
            "bio": bio,
            "photo_url": photo_url,  # Use Cloudinary URL
            "password": hashed_password,
            "email":email,
            "role": "user"
        }

        # Debug: Log user data
        print("User data to insert:", user_data)

        # Insert user data into MongoDB
        try:
            collection.insert_one(user_data)
            print("User data inserted successfully")
        except pymongo.errors.DuplicateKeyError as e:
            print("Duplicate key error:", str(e))
            return jsonify({"success": False, "error": "User name already exists"}), 400
        except Exception as e:
            print("MongoDB insertion failed:", str(e))
            return jsonify({"success": False, "error": "Failed to insert user data into MongoDB"}), 500

        # Set user session
        session['user_name'] = user_name
        session['user_role'] = 'user'

        # Return success response
        return jsonify({"success": True, 'name': name})

    except Exception as e:
        # Handle any other exceptions
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)
        return jsonify({"success": False, "error": error_message}), 500
@app.route("/delete_user/<user_id>", methods=['POST'])
def delete_user(user_id):
    # Check if the session is for an admin
    if 'admin_name' not in session or session.get('admin_role') != 'admin':
        return redirect(url_for('admin_login_page'))

    collection.delete_one({"_id": ObjectId(user_id)})
    return redirect(url_for('admin'))
@app.route("/edit_user/<user_id>", methods=['GET', 'POST'])
def edit_user(user_id):
    # Check if the session is for an admin
    if 'admin_name' not in session or session.get('admin_role') != 'admin':
        return redirect(url_for('admin_login_page'))

    # Fetch the user from the database
    user = collection.find_one({"_id": ObjectId(user_id)})
    
    if request.method == 'POST':
        # Get form data
        name = request.form.get("name")
        rollno = request.form.get("rollno")
        branch = request.form.get("branch")
        registrationno = request.form.get("registrationno")
        bio = request.form.get("bio")
        email = request.form.get("email")
        photo = request.files.get('photo')

        # Initialize photo_url with the existing photo URL
        photo_url = user['photo_url']

        # Handle photo upload if a new photo is provided
        if photo:
            try:
                cloudinary_response = cloudinary.uploader.upload(photo)
                photo_url = cloudinary_response['secure_url']
                print("Cloudinary response:", cloudinary_response)
            except Exception as e:
                print("Cloudinary upload failed:", str(e))
                return jsonify({"success": False, "error": "Failed to upload photo to Cloudinary"}), 500

        # Validate and convert rollno and registrationno to integers
        try:
            rollno = int(rollno)
            registrationno = int(registrationno)
        except ValueError:
            return jsonify({"success": False, "error": "rollno and registration_no must be integers"}), 400

        # Update the user's details in the database
        collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "name": name,
                "rollno": rollno,
                "branch": branch,
                "registration_no": registrationno,
                "bio": bio,
                "photo_url": photo_url,
                "email": email,
                "role": "user"
            }}
        )

        # Redirect to the admin dashboard after updating the user
        flash("User details updated successfully!", "success")
        return redirect(url_for('admin'))

    # Render the edit user form with the user's current data
    return render_template('edit_user.html', user=user)
@app.route('/user_login')
def user_login():
    return render_template('login.html')
@app.route("/login", methods=["POST"])
def login():
    user_name = request.form.get("user_name")
    password = request.form.get("password")
    

    if not user_name or not password:
        return jsonify({"success": False, "error": "Name and password are required"}), 400

    # Find the user in the database
    user = collection.find_one({"user_name": user_name})
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    # Verify the password
    if not check_password_hash(user.get("password", ""), password):
        return jsonify({"success": False, "error": "Incorrect password"}), 401

    # Set session for the logged-in user
    session['user_name'] = user['user_name']
    session["email"]= user["email"]
    session['user_role'] = 'user'  # Add role for clarity
    return jsonify({"success": True, "user_name": user['user_name']})
@app.route("/user_login_face", methods=["GET"])
def user_login_face():
    return render_template("user_login_face.html")
@app.route("/login_face", methods=["POST"])
def login_face():
    # Get form data
    name = request.form.get("name")
    photo = request.files.get("photo")

    # Validate input
    if not name or not photo:
        return jsonify({"success": False, "error": "Name and photo are required"}), 400

    # Save the login photo to the uploads directory
    filename = secure_filename(photo.filename)
    login_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    photo.save(login_photo_path)

    # Load and encode the login photo
    login_image = face_recognition.load_image_file(login_photo_path)
    login_face_encodings = face_recognition.face_encodings(login_image)
    if len(login_face_encodings) == 0:
        return jsonify({"success": False, "error": "No face found in the login image"}), 400

    login_face_encoding = login_face_encodings[0]

    # Iterate through all users in the database
    for user in collection.find({"role": "user"}):
        # Fetch the user's photo URL from Cloudinary
        photo_url = user.get("photo_url")
        if not photo_url:
            continue  # Skip users without a photo URL

        # Download the image from Cloudinary
        response = requests.get(photo_url)
        if response.status_code != 200:
            continue  # Skip if the image cannot be fetched

        # Load the registered user's photo
        registered_image = Image.open(BytesIO(response.content))
        registered_image = np.array(registered_image)  # Convert to numpy array for face_recognition

        # Encode the registered user's face
        registered_face_encodings = face_recognition.face_encodings(registered_image)
        if len(registered_face_encodings) == 0:
            continue  # Skip if no face is found in the registered image

        registered_face_encoding = registered_face_encodings[0]


        # Compare faces
        matches = face_recognition.compare_faces([registered_face_encoding], login_face_encoding, tolerance=0.5)
        print("Face matches:", matches)

        if any(matches):
            # If faces match, create a session for the user
            session['user_name'] = user['user_name']
            session['user_role'] = 'user'
            return jsonify({
                "success": True,
                "name": user['name']
            })

    # If no matching user is found
    return jsonify({"success": False, "error": "No matching user found"}), 404
@app.route("/google_user_profile")
def google_profile():
    # Check if the session is for a user
    if 'email' not in session:
        return redirect(url_for('user_login'))
    jobs=get_jobs_from_db()
    email= session['email']
    user = collection.find_one({"email": email})
    job_applications = list(job_applications_collection.find({"user_name":user["user_name"]}))

    # Add job applications to the user object
    user['application_status'] = job_applications
    return render_template("success.html", user=user,jobs=jobs)
@app.route("/user_profile")
def user_profile():
    # Check if the session is for a user
    if 'user_name' not in session or session.get('user_role') != 'user':
        return redirect(url_for('user_login'))
    jobs=get_jobs_from_db()
    user_name = session['user_name']
    user = collection.find_one({"user_name": user_name})
    job_applications = list(job_applications_collection.find({"user_name": user_name}))

    # Add job applications to the user object
    user['application_status'] = job_applications
    return render_template("success.html", user=user,jobs=jobs)

@app.route("/admin_success/<user_id>")
def admin_success(user_id):
    # Check if the session is for an admin
    if 'admin_name' not in session or session.get('admin_role') != 'admin':
        return redirect(url_for('admin_login_page'))

    # Fetch the user's details from the database
    user = collection.find_one({"_id": ObjectId(user_id)})

    if not user:
        return redirect(url_for('user_login'))
    
  # Redirect to login if user not found
    job_applications = list(job_applications_collection.find({"user_name": user["user_name"]}))

    # Add job applications to the user object
    user['application_status'] = job_applications
    return render_template("success.html", user=user)

@app.route("/logout")
def logout():
    # Clear user session keys
    session.pop('user_name', None)
    session.pop('email', None)
    session.pop('user_role', None)
    return redirect(url_for('index'))

@app.route("/admin_logout")
def admin_logout():
    # Clear user session keys
    session.pop('admin_name', None)
    session.pop('admin_role', None)
    return redirect(url_for('index'))
GENERATED_UPLOAD_FOLDER = 'static/generated_uploads/'
app.config['GENERATED_UPLOAD_FOLDER'] = GENERATED_UPLOAD_FOLDER
os.makedirs(GENERATED_UPLOAD_FOLDER, exist_ok=True)

UPDATE_UPLOAD_FOLDER = 'static/update_uploads/'
app.config['UPDATE_UPLOAD_FOLDER'] = UPDATE_UPLOAD_FOLDER
os.makedirs(UPDATE_UPLOAD_FOLDER, exist_ok=True)

RESUME_FOLDER = 'static/resume_uploads/'
os.makedirs(RESUME_FOLDER, exist_ok=True)
app.config['RESUME_FOLDER'] =RESUME_FOLDER


nlp = spacy.load("en_core_web_sm")  # Load NLP model
def fetch_admin_jobs():
    """Fetches job descriptions and skills from the database."""
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM job_descriptions")
    jobs = cursor.fetchall()
    conn.close()
    return jobs
# Fetch job descriptions from the database
import sqlite3

def fetch_jobs():
    """Fetches job descriptions and skills from the database."""
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT j.title, j.description, j.skills, c.name, c.logo_url
        FROM job_descriptions j
        JOIN companies c ON j.company_id = c.id
    """)
    jobs = [{"title": row[0], "description": row[1], "skills": row[2], "company": row[3], "logo_url": row[4]} for row in cursor.fetchall()]
    conn.close()
    return jobs


SOFTWARE_SKILLS = {
    'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin',
    'html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'node.js', 
    'git', 'docker', 'kubernetes', 'aws', 'azure', 'sql', 'mongodb', 'postgresql',
    'machine learning', 'deep learning', 'data analysis', 'artificial intelligence',
    'rest api', 'graphql', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas',
    'numpy', 'linux', 'agile', 'scrum', 'devops', 'cybersecurity', 'blockchain',
    'big data', 'hadoop', 'spark', 'nosql', 'oop', 'functional programming',
    'unit testing', 'ci/cd', 'microservices', 'serverless', 'redux', 'typescript',
    'spring boot', '.net', 'laravel', 'rails', 'express.js', 'keras', 'ansible',
    'terraform', 'jenkins', 'golang', 'rust', 'elasticsearch', 'rabbitmq', 'kafka'
}

def extract_skills(text):
    """Extracts software-related skills from text using NLP and predefined patterns."""
    doc = nlp(text.lower())  # Process text in lowercase for consistent matching
    skills = set()

    # Check noun chunks for multi-word skills
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.replace(' ', '_')  # Format multi-word phrases
        if chunk_text in SOFTWARE_SKILLS:
            skills.add(chunk_text.replace('_', ' '))

    # Check individual tokens
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
            lemma = token.lemma_.lower()
            if lemma in SOFTWARE_SKILLS or token.text.lower() in SOFTWARE_SKILLS:
                skills.add(token.text.lower())

    # Additional pattern matching for version numbers and special cases
    for match in re.finditer(r'\b(?:python|java|c\+\+|c#|\.net)\s*\d*\.?\d+\b', text.lower()):
        skills.add(match.group().strip())

    # Handle common variations
    variations_map = {
        'js': 'javascript',
        'node': 'nodejs',
        'reactjs': 'react',
        'vuejs': 'vue',
        'angularjs': 'angular',
        'ml': 'machine learning',
        'ai': 'artificial intelligence',
        'aws cloud': 'aws',
        "sql": "mysql"
    }

    for variation, standard in variations_map.items():
        if variation in skills:
            skills.remove(variation)
            skills.add(standard)

    return skills

def calculate_ats_score(resume_text, job_description, resume_skills, job_skills):
    """Calculates ATS score using text similarity and skill matching."""
    # Convert resume_skills and job_skills to sets
    resume_skills_set = set(resume_skills)
    job_skills_set = set(job_skills)

    # Calculate text similarity using TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer().fit_transform([resume_text, job_description])
    similarity_score = cosine_similarity(vectorizer[0], vectorizer[1])[0][0] * 100

    # Calculate skill match score
    matched_skills = resume_skills_set.intersection(job_skills_set)
    skill_match_score = (len(matched_skills) / len(job_skills_set)) * 100 if job_skills_set else 0

    # Calculate overall ATS score
    ats_score = round((similarity_score * 0.7) + (skill_match_score * 0.3), 2)
    return ats_score, list(job_skills_set - matched_skills)
def recommend_jobs(resume_text, resume_skills):
    jobs = fetch_jobs()
    job_scores = []

    for job in jobs:
        title, description, skills = job["title"], job["description"], job["skills"]
        job_skills = set(skills.lower().split(", "))  # Convert stored skills into a set
        ats_score, missing_skills = calculate_ats_score(resume_text, description, resume_skills, job_skills)
        job_scores.append({
            "title": title,
            "ats_score": ats_score,
            "missing_skills": missing_skills
        })

    # Debugging: Print the job scores to see their values
    

    # Ensure ATS score is numeric and sort jobs by score in descending order
    job_scores.sort(key=lambda x: x['ats_score'], reverse=True)

    # Debugging: Print the sorted job scores

    # Return top 3 jobs based on ATS score
    return job_scores[:3]
def extract_text_from_pdf(pdf_input):
    """Extract text from PDF."""
    text = ""
    try:
        if isinstance(pdf_input, BytesIO):
            pdf_input.seek(0)  # Reset the stream position to the beginning
            reader = PyPDF2.PdfReader(pdf_input)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += " " + page_text
        else:
            raise ValueError("Invalid input type. Expected BytesIO.")
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    return text.strip()

# Flask Routes
@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    return send_from_directory(app.config['GENERATED_UPLOAD_FOLDER'], filename, as_attachment=True, mimetype='application/pdf')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if 'email' not in session:
        return redirect(url_for('user_login'))
    if 'user_name' not in session or session.get('user_role') != 'user':
        return jsonify({"success": False, "error": "User not logged in"}), 401

    user_name = session['user_name']
    file = request.files.get('resume')

    if not file or file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type. Only PDF allowed."}), 400

    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    file_content = file.read()
    if len(file_content) > MAX_FILE_SIZE:
        return jsonify({"success": False, "error": "File size exceeds 5MB limit"}), 400

    if not file_content:
        return jsonify({"success": False, "error": "Uploaded file is empty"}), 400

    try:
        file.seek(0)  # Reset file pointer to beginning before storing
        file_id = fs.put(file, filename=secure_filename(file.filename), content_type="application/pdf")
        print(f"File uploaded successfully with ID: {file_id}")  # Log file ID for debugging

        file_stream = BytesIO(file_content)
        resume_text = extract_text_from_pdf(file_stream)
        if not resume_text:
            return jsonify({"success": False, "error": "Failed to extract text from the resume"}), 400

        resume_skills = list(extract_skills(resume_text))
        

        job_recommendations = recommend_jobs(resume_text, resume_skills)
        ats_score, missing_skills = ("N/A", [])
        print("ats_score:",ats_score)
        print("missing_skill:",missing_skills)

        if job_recommendations and len(job_recommendations) > 0:
            job = job_recommendations[0]
            print(job)
            ats_score, missing_skills =job["ats_score"],job["missing_skills"]
        print("ats_score:",ats_score)
        print("missing_skill:",missing_skills)

        user_resume_data = {
            "resume_pdf_id": file_id,
            "upload_text": resume_text,
            "extracted_skills": resume_skills,
            "ats_score": ats_score,
            "missing_skills": missing_skills,
            "job_recommendations": job_recommendations,
            "upload_date": datetime.now().isoformat()
        }

        print(f"Updating user resume data for {user_name}")  # Log user data update attempt
        collection.update_one(
            {"user_name": user_name},
            {"$set": user_resume_data},
            upsert=True
        )

        return jsonify({
            "success": True,
            "message": "Resume uploaded successfully!",
            "resume_text": resume_text,
            "ats_score": ats_score,
            "missing_skills": missing_skills,
            "job_recommendations": job_recommendations,
            "download_url": url_for('download_resume', pdf_id=str(file_id), _external=True),
            "profile_url": url_for('view_profile', _external=True)
        })

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"success": False, "error": f"An error occurred while processing the resume: {str(e)}"}), 500

import time
def upload_to_cloudinary(image_data, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = cloudinary.uploader.upload(image_data, resource_type="image")
            return response
        except CloudinaryError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise  # Re-raise the exception after all retries fail
@app.route("/submit", methods=["POST"])
def submit():
    if 'email' not in session:
        return redirect(url_for('user_login'))
    if 'user_name' not in session:
        return jsonify({"error": "User not logged in"}), 401

    user_name = session['user_name']
    data = request.json

    try:
        # Decode image data
        image_data = base64.b64decode(data['photo'].split(",")[1])
        print("Image data length:", len(image_data))  # Debug: Check the size of the image data

        # Upload photo to Cloudinary
        try:
            photo_response = upload_to_cloudinary(image_data)
            photo_url = photo_response['secure_url']
        except CloudinaryError as e:
            return jsonify({"error": f"Failed to upload photo to Cloudinary: {e}"}), 500

        # Generate PDF
        pdf_filename = f"resume_{data['name']}.pdf"
        pdf_path = os.path.join(app.config['GENERATED_UPLOAD_FOLDER'], pdf_filename)
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"Name: {data['name']}", styles['Title']),
            Spacer(1, 0.2 * inch),
            Paragraph(f"Email: {data['email']}", styles['BodyText']),
            Paragraph(f"Phone: {data['phone']}", styles['BodyText']),
            Paragraph(f"Skills: {', '.join(data['skills'])}", styles['BodyText']),
            Paragraph(f"Summary: {data['summary']}", styles['BodyText']),
            Paragraph("Experience:", styles['Heading2']),
            Paragraph(data['experience'], styles['BodyText']),
            Paragraph("Education:", styles['Heading2']),
            Paragraph(data['education'], styles['BodyText'])
        ]
        doc.build(story)

        # Debug: Verify PDF file exists and is not empty
        if os.path.exists(pdf_path):
            print(f"PDF file created at: {pdf_path}")
            print(f"File size: {os.path.getsize(pdf_path)} bytes")
        else:
            print("PDF file not created!")
            return jsonify({"error": "Failed to generate PDF."}), 500

        # Extract text from PDF
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes_io = BytesIO(pdf_file.read())  # Convert file to BytesIO
        resume_text = extract_text_from_pdf(pdf_bytes_io)
        resume_skills = list(data['skills']) if 'skills' in data else []

        job_recommendations = recommend_jobs(resume_text, resume_skills)
        ats_score, missing_skills = ("N/A", [])

        if job_recommendations and len(job_recommendations) > 0:
            job = job_recommendations[0]
            print(job)
            ats_score, missing_skills = job["ats_score"], job["missing_skills"]
        print("ats_score:", ats_score)
        print("missing_skill:", missing_skills)

        # Update user data in MongoDB
        try:
            user=collection.find_one({"user_name":user_name})
            collection.update_one(
                {"user_name": user_name},
                {"$set": {
                    "name": data['name'],
                    "email": data['email'],
                    "phone": data['phone'],
                    "skills": list(resume_skills),
                    "summary": data['summary'],
                    "experience": data['experience'],
                    "education": data['education'],
                    "resume_path": pdf_path,  # Save the local file path
                    "photo_url": photo_url,
                    "resume_text": resume_text,
                    "ats_score": ats_score,
                    "missing_skills": list(missing_skills),
                    "generated_job_recommendations": job_recommendations,
                    "upload_date": ist_time_str
                }},
                upsert=True
            )
            return jsonify({
                "message": "Resume generated successfully!",
                "resume_text": resume_text,
                "ats_score": ats_score,
                "missing_skills": missing_skills,
                "job_recommendations": job_recommendations,
                "download_url": url_for('Generate_download_resume', user_id=user["_id"], _external=True),  # Generate download URL
                "profile_url": url_for('view_profile', _external=True)
            })
        except Exception as e:
            print(f"MongoDB Error: {e}")
            return jsonify({"error": "An error occurred while saving the resume."}), 500

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500
from flask import send_file

@app.route('/Generate_download_resume/<user_id>')
def Generate_download_resume(user_id):
    """
    Serves the PDF file directly from the server.
    """
    try:
        # Fetch the user's resume path from the database
        user = collection.find_one({"_id": ObjectId(user_id)})
        if not user or "resume_path" not in user:
            return jsonify({"error": "Resume not found"}), 404

        pdf_path = user["resume_path"]

        # Serve the file for download
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"resume_{user['name']}.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        print(f"Error serving file: {e}")
        return jsonify({"error": "Failed to download resume"}), 500
@app.route('/download_resume/<pdf_id>')
def download_resume(pdf_id):
    """Serves the resume file from GridFS."""
    try:
        file = fs.get(ObjectId(pdf_id))
        if not file:
            return jsonify({"error": "File not found in GridFS"}), 404
        
        return send_file(
            BytesIO(file.read()),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=file.filename
        )
    except Exception as e:
        print(f"Error serving file: {e}")
        return jsonify({"error": "File not found"}), 404
@app.route('/view_resume/<user_id>')  # Example route
def view_resume(user_id):
    user = collection.find_one({"_id": ObjectId(user_id)})
    return render_template('view_resume.html', user=user)
@app.route('/view_resume_upload/<resume_pdf_id>')  # Example route
def view_resume_upload(resume_pdf_id):
    user = collection.find_one({"resume_pdf_id": ObjectId(resume_pdf_id)})
    return render_template('view_resume.html', user=user)

@app.route("/resume_delete", methods=["POST"])
def resume_delete():
    if 'user_name' not in session:
        return jsonify({"error": "User not logged in"}), 401

    user_name = session['user_name']
    user = collection.find_one({"user_name": user_name})

    if not user:
        return jsonify({"error": "User not found"}), 404

    try:
        # 1. Delete from GridFS (if applicable)
        if user.get('resume_pdf_id'):
            fs.delete(ObjectId(user['resume_pdf_id']))

        # 2. Update user record in MongoDB
        collection.update_one(
            {"user_name": user_name},
            {"$unset": {
                "resume_pdf_id": "",
                "upload_text": "",
                "extracted_skills": "",
                "ats_score": "",
                "missing_skills": "",
                "job_recommendations": ""
            }}
        )

        flash("Resume deleted successfully!", "success")
        return redirect(url_for('user_profile'))  # Redirect to the success page

    except Exception as e:
        print(f"Error deleting resume: {e}")
        flash("An error occurred while deleting the resume.", "error")
        return redirect(url_for('user_profile'))
@app.route("/Generated_resume_delete", methods=["POST"])
def Generated_resume_delete():
    if 'user_name' not in session:
        return jsonify({"error": "User not logged in"}), 401

    user_name = session['user_name']
    user = collection.find_one({"user_name": user_name})

    if not user:
        return jsonify({"error": "User not found"}), 404

    try:

        collection.update_one(
            {"user_name": user_name},
            {"$unset": {
                "email": "",
                    "phone": "",
                    "skills": "",
                    "summary": "",
                    "experience": "",
                    "education": "",
                    "resume_path":"", 
                    "resume_text": "",
                    "ats_score": "",
                    "missing_skills": "",
                    "generated_job_recommendations": ""
            }}
        )

        flash("Resume deleted successfully!", "success")
        return redirect(url_for('success'))  # Redirect to the success page

    except Exception as e:
        print(f"Error deleting resume: {e}")
        flash("An error occurred while deleting the resume.", "error")
        return redirect(url_for('success'))
@app.route('/view_profile')
def view_profile():
    # Check if the session is for a user
    return redirect('user_profile')
@app.route('/admin_view_profile/<user_id>')
def admin_view_profile(user_id):
    # Check if the session is for a user
    if 'admin_name' not in session or session.get('admin_role') != 'admin':
        return jsonify({"error": "User not logged in"}), 401

    
    user = collection.find_one({"_id": ObjectId(user_id)})

    if not user:
        return jsonify({"error": "User not found"}), 404

    return render_template('success.html', user=user)


@app.route("/update_resume", methods=["GET", "POST"])
def update_resume():
    if 'user_name' not in session:
        return jsonify({"error": "User not logged in"}), 401

    user_name = session['user_name']

    if request.method == "GET":
        user = collection.find_one({"user_name": user_name})
        if not user:
            return jsonify({"error": "User not found"}), 404
        return render_template("update_resume.html", user=user)

    elif request.method == "POST":
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        # Upload photo to Cloudinary
        image_data = base64.b64decode(data['photo'].split(",")[1])
        photo_response = cloudinary.uploader.upload(image_data, resource_type="image")
        photo_url = photo_response['secure_url']

        # Generate PDF
        pdf_filename = f"resume_{data['name']}.pdf"
        pdf_path = os.path.join(app.config['UPDATE_UPLOAD_FOLDER'], pdf_filename)
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"Name: {data['name']}", styles['Title']),
            Spacer(1, 0.2 * inch),
            Paragraph(f"Email: {data['email']}", styles['BodyText']),
            Paragraph(f"Phone: {data['phone']}", styles['BodyText']),
            Paragraph(f"Skills: {', '.join(data['skills'])}", styles['BodyText']),
            Paragraph(f"Summary: {data['summary']}", styles['BodyText']),
            Paragraph("Experience:", styles['Heading2']),
            Paragraph(data['experience'], styles['BodyText']),
            Paragraph("Education:", styles['Heading2']),
            Paragraph(data['education'], styles['BodyText'])
        ]
        doc.build(story)

        # Upload PDF to Cloudinary
        pdf_response = cloudinary.uploader.upload(pdf_path, resource_type="raw")
        pdf_url = pdf_response['secure_url']
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes_io = BytesIO(pdf_file.read())  
        resume_text = extract_text_from_pdf(pdf_bytes_io)
        resume_skills = data.get('skills', [])
        job_recommendations = recommend_jobs(resume_text, resume_skills)
        ats_score, missing_skills = ("N/A", [])

        if job_recommendations and len(job_recommendations) > 0:
            job = job_recommendations[0]
            print(job)
            ats_score, missing_skills =job["ats_score"],job["missing_skills"]
        print("ats_score:",ats_score)
        print("missing_skill:",missing_skills)

        updated_resume_data = {
            "resume_url": pdf_url,
            "pdf_path":pdf_path,  # Use Cloudinary URL
            "resume_text": resume_text,
            "extracted_skills": list(resume_skills),
            "ats_score": ats_score,
            "missing_skills": list(missing_skills),
            "job_recommendations": job_recommendations,
            "upload_date": datetime.now().isoformat(),
            "photo_url": photo_url,
            "name": data.get('name'),
            "email": data.get('email'),
            "phone": data.get('phone'),
            "summary": data.get('summary'),
            "experience": data.get('experience'),
            "education": data.get('education')
        }

        try:
            collection.update_one({"user_name": user_name}, {"$set": updated_resume_data})
            return jsonify({
                "message": "Resume updated successfully!",
                "resume_text": resume_text,
                "ats_score": ats_score,
                "missing_skills": missing_skills,
                "job_recommendations": job_recommendations,
                "download_url": pdf_url,  # Use Cloudinary URL
                "profile_url": url_for("view_profile", _external=True),
            })
        except Exception as e:
            print(f"MongoDB Error: {e}")
            return jsonify({"error": "An error occurred while updating the resume."}), 500

@app.route("/get_admin_update_resume/<user_id>", methods=["GET"])
def get_admin_update_resume(user_id):
          # Get user_id from query parameters
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        user = collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404

        return render_template("admin_update_resume.html", user=user)
@app.route("/admin_update_resume/<user_id>", methods=["POST"])
def admin_update_resume(user_id):
    if 'admin_name' not in session:
        return jsonify({"error": "Admin not logged in"}), 401

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    # Upload photo to Cloudinary
    image_data = base64.b64decode(data['photo'].split(",")[1])
    photo_response = cloudinary.uploader.upload(image_data, resource_type="image")
    photo_url = photo_response['secure_url']

    # Generate PDF
    pdf_filename = f"resume_{data['name']}.pdf"
    pdf_path = os.path.join(app.config['UPDATE_UPLOAD_FOLDER'], pdf_filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"Name: {data['name']}", styles['Title']),
        Spacer(1, 0.2 * inch),
        Paragraph(f"Email: {data['email']}", styles['BodyText']),
        Paragraph(f"Phone: {data['phone']}", styles['BodyText']),
        Paragraph(f"Skills: {', '.join(data['skills'])}", styles['BodyText']),
        Paragraph(f"Summary: {data['summary']}", styles['BodyText']),
        Paragraph("Experience:", styles['Heading2']),
        Paragraph(data['experience'], styles['BodyText']),
        Paragraph("Education:", styles['Heading2']),
        Paragraph(data['education'], styles['BodyText'])
    ]
    doc.build(story)

    # Upload PDF to Cloudinary
    pdf_response = cloudinary.uploader.upload(pdf_path, resource_type="raw")
    pdf_url = pdf_response['secure_url']
    with open(pdf_path, "rb") as pdf_file:
            pdf_bytes_io = BytesIO(pdf_file.read())
    resume_text = extract_text_from_pdf(pdf_bytes_io)
    resume_skills = data.get('skills', [])
    job_recommendations = recommend_jobs(resume_text, resume_skills)
    ats_score, missing_skills = ("N/A", [])

    if job_recommendations and len(job_recommendations) > 0:
            job = job_recommendations[0]
            print(job)
            ats_score, missing_skills =job["ats_score"],job["missing_skills"]
    print("ats_score:",ats_score)
    print("missing_skill:",missing_skills)
    

    updated_resume_data = {
        "resume_url": pdf_url,
        "pdf_path":pdf_path,  # Use Cloudinary URL
        "resume_text": resume_text,
        "extracted_skills": list(resume_skills),
        "ats_score": ats_score,
        "missing_skills": list(missing_skills),
        "job_recommendations": job_recommendations,
        "upload_date": datetime.datetime.utcnow(),
        "photo_url": photo_url,
        "name": data.get('name'),
        "email": data.get('email'),
        "phone": data.get('phone'),
        "summary": data.get('summary'),
        "experience": data.get('experience'),
        "education": data.get('education')
    }

    try:
        collection.update_one({"_id": ObjectId(user_id)}, {"$set": updated_resume_data})
        return jsonify({
            "message": "Resume updated successfully!",
            "resume_text": resume_text,
            "ats_score": ats_score,
            "missing_skills": missing_skills,
            "job_recommendations": job_recommendations,
            "download_url": pdf_url,  # Use Cloudinary URL
            "profile_url": url_for("admin_view_profile", user_id=user_id, _external=True),
        })
    except Exception as e:
        print(f"MongoDB Error: {e}")
        return jsonify({"error": "An error occurred while updating the resume."}), 500
@app.route('/download_resume/<path:pdf_url>')
def generate_download_resume(pdf_url):
    """Serves the resume file from Cloudinary."""
    try:
        # Fetch the file from Cloudinary
        response = cloudinary.api.resource(pdf_url, resource_type="raw")
        if not response:
            return jsonify({"error": "File not found in Cloudinary"}), 404
        
        # Redirect to the secure URL for download
        return redirect(response['secure_url'])
    
    except CloudinaryError as e:
        print(f"Cloudinary Error: {e}")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500
@app.route('/download_photo/<photo_url>')
def download_photo(photo_url):
    """Serves the photo file from Cloudinary."""
    try:
        # Fetch the file from Cloudinary
        response = cloudinary.api.resource(photo_url, resource_type="image")
        if not response:
            return jsonify({"error": "File not found in Cloudinary"}), 404
        
        # Get the secure URL of the file
        file_url = response['secure_url']
        
        # Redirect to the file URL for download
        return redirect(file_url)
    
    except CloudinaryError as e:
        print(f"Cloudinary Error: {e}")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

from urllib.parse import unquote
@app.route("/apply_job_user", methods=["GET"])
def apply_job_user():
    jobs=get_jobs_from_db()
    return render_template("apply.html",jobs=jobs)
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

def send_confirmation_email(user_email, job_title, company_name, logo_url):
    """Send a confirmation email to the user with company logo and name."""
    subject = "Job Application Confirmation"
    
    # HTML email body with company logo and name
    html_body = f"""
    <html>
    <body>
        <div style="text-align: center;">
            <img src="{logo_url}" alt="{company_name} Logo" style="width: 100px; height: auto;">
            <h2>{company_name}</h2>
        </div>
        <p>Dear Applicant,</p>
        <p>Thank you for applying for the position of <strong>{job_title}</strong> at <strong>{company_name}</strong>.</p>
        <p>Your application has been successfully submitted.</p>
        <p>Due to the high volume of applications, we will carefully review your profile and sincerely consider you for applicable roles.</p>
        <p>Best regards,</p>
        <p>The Hiring Team</p>
    </body>
    </html>
    """

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = SMTP_EMAIL
    msg['To'] = user_email
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))  # Attach HTML content

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, user_email, msg.as_string())
        print("Confirmation email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


def send_selection_email(user_email, job_title, company_name, logo_url):
    """Send an email to notify the user that their application has been accepted."""
    subject = "Congratulations! Your Application Has Been Accepted"
    
    # HTML email body with company logo and name
    html_body = f"""
    <html>
    <body>
        <div style="text-align: center;">
            <img src="{logo_url}" alt="{company_name} Logo" style="width: 100px; height: auto;">
            <h2>{company_name}</h2>
        </div>
        <p>Dear Applicant,</p>
        <p>We are pleased to inform you that your application for the position of <strong>{job_title}</strong> at <strong>{company_name}</strong> has been accepted. Congratulations!</p>
        <p>Our team will contact you shortly to discuss the next steps in the hiring process.</p>
        <p>Best regards,</p>
        <p>The Hiring Team</p>
    </body>
    </html>
    """

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = SMTP_EMAIL
    msg['To'] = user_email
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))  # Attach HTML content

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, user_email, msg.as_string())
        print("Selection email sent successfully!")
    except Exception as e:
        print(f"Error sending selection email: {e}")


def send_rejection_email(user_email, job_title, company_name, logo_url):
    """Send an email to notify the user that their application has been rejected."""
    subject = "Application Status Update"
    
    # HTML email body with company logo and name
    html_body = f"""
    <html>
    <body>
        <div style="text-align: center;">
            <img src="{logo_url}" alt="{company_name} Logo" style="width: 100px; height: auto;">
            <h2>{company_name}</h2>
        </div>
        <p>Dear Applicant,</p>
        <p>Thank you for applying for the position of <strong>{job_title}</strong> at <strong>{company_name}</strong>.</p>
        <p>After careful consideration, we regret to inform you that your application has not been selected for further processing.</p>
        <p>We appreciate your interest in our organization and encourage you to apply for future opportunities that match your skills and experience.</p>
        <p>Best regards,</p>
        <p>The Hiring Team</p>
    </body>
    </html>
    """

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = SMTP_EMAIL
    msg['To'] = user_email
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))  # Attach HTML content

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, user_email, msg.as_string())
        print("Rejection email sent successfully!")
    except Exception as e:
        print(f"Error sending rejection email: {e}")

from datetime import datetime
import pytz

# Get the current time in IST
ist_timezone = pytz.timezone('Asia/Kolkata')
ist_time = datetime.now(ist_timezone)

# Convert IST time to a string
ist_time_str = ist_time.strftime('%I:%M %p %d-%m-%Y')

@app.route("/apply_job/<int:job_id>", methods=["POST"])
def apply_job(job_id):
    # Decode the job_title to handle spaces and special characters
    print(f"Job title received: {job_id}")

    if 'user_name' not in session or session.get('user_role') != 'user':
        return jsonify({"error": "User not logged in"}), 401

    user_name = session['user_name']
    user = collection.find_one({"user_name": user_name})

    if not user:
        return jsonify({"error": "User not found"}), 404

    # Fetch the job details from the database using job_title
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT j.title, c.name ,c.logo_url FROM job_descriptions j JOIN companies c ON j.company_id = c.id WHERE j.id = ?", (job_id,))
    job = cursor.fetchone()
    conn.close()

    if not job:
        return jsonify({"error": "Job not found"}), 404

    job_title, company_name, logo_url = job  # Extract job title and company name

    # Check if the user has already applied for this job
    existing_application = job_applications_collection.find_one({
        "user_name": user_name,
        "job_title": job_title
    })

    if existing_application:
        return jsonify({"error": "You have already applied for this job"}), 400

    # Use the user's existing resume or generate a new one
    if user.get('resume_pdf_id'):
        resume_pdf_id = user['resume_pdf_id']
        resume_text = user.get('upload_text', '')
    else:
        return jsonify({"error": "No resume found. Please upload resume first."}), 400

    # Save the job application
    application_data = {
        "user_name": user_name,
        "job_title": job_title,
        "company_name": company_name,
        "logo_url":logo_url,  # Add company name to the application data
        "resume_pdf_id": resume_pdf_id,
        "resume_text": resume_text,
        "ats_score": user["ats_score"],
        "missing_skills": user["missing_skills"],
        "status": "pending",  # Initial status
        "application_date": ist_time_str
    }
    job_applications_collection.insert_one(application_data)
    collection.update_one(
        {"user_name": user_name},
        {"$set": {"application_status": list(application_data)}}
    )

    # Send confirmation email with job title and company name
    try:
        send_confirmation_email(user['email'], job_title, company_name,logo_url)
    except Exception as e:
        print(f"Failed to send confirmation email: {e}")

    return jsonify({
        "message": "Job application submitted successfully!",
        "ats_score": user["ats_score"],
        "missing_skills": user["missing_skills"],
        "status": "pending"
    })
@app.route("/update_application_status/<application_id>", methods=["POST"])
def update_application_status(application_id):
    # Check if the admin is logged in
    if 'admin_name' not in session or session.get('admin_role') != 'admin':
        return jsonify({"error": "Admin not logged in"}), 401

    # Validate the request data
    data = request.get_json()
    if not data or 'status' not in data:
        return jsonify({"error": "Status is required"}), 400

    status = data['status']
    if status not in ["accepted", "rejected"]:
        return jsonify({"error": "Invalid status"}), 400

    try:
        # Fetch the application to get the user_name, job_title, and company_name
        application = job_applications_collection.find_one({"_id": ObjectId(application_id)})
        if not application:
            return jsonify({"error": "Application not found"}), 404

        user_name = application.get("user_name")
        job_title = application.get("job_title")
        company_name = application.get("company_name")
        logo_url= application.get("logo_url") # Fetch company name
        if not user_name or not job_title or not company_name:
            return jsonify({"error": "User or job details not found in application"}), 404

        # Fetch the user's email from the users collection
        user = collection.find_one({"user_name": user_name})
        if not user:
            return jsonify({"error": "User not found"}), 404

        user_email = user.get("email")
        if not user_email:
            return jsonify({"error": "User email not found"}), 404

        # Update the application status in job_applications_collection
        job_applications_collection.update_one(
            {"_id": ObjectId(application_id)},
            {"$set": {"status": status}}
        )

        # Update the user's document in the collection (users database)
        collection.update_one(
            {"user_name": user_name},
            {"$set": {"application_status": application}}
        )

        # Send email notification based on status
        if status == "accepted":
            send_selection_email(user_email, job_title, company_name,logo_url)
        elif status == "rejected":
            send_rejection_email(user_email, job_title, company_name,logo_url)

        return jsonify({"message": f"Application status changed to {status}"})

    except Exception as e:
        print(f"Error updating application status: {e}")
        return jsonify({"error": "An error occurred while updating the application status"}), 500
@app.route("/view_applications", methods=["GET"])
def view_applications():
    if 'admin_name' not in session or session.get('admin_role') != 'admin':
        return jsonify({"error": "Admin not logged in"}), 401

    applications = list(job_applications_collection.find({}))

    return render_template("view_applications.html", applications=applications)
@app.route("/delete_application/<application_id>", methods=["DELETE"])
def admin_delete_application(application_id):
    application = job_applications_collection.find_one({
        "_id": ObjectId(application_id),
    })
    if not application:
        return jsonify({"error": "Application not found or unauthorized"}), 404

    job_applications_collection.delete_one({"_id":ObjectId(application_id)})
    return jsonify({"message": "Application deleted successfully!"})
@app.route("/user_delete_application", methods=["DELETE"])
def user_delete_application():
    if 'user_name' not in session or session.get('user_role') != 'user':
        return jsonify({"error": "User not logged in"}), 401
    user_name = session['user_name']
    job_applications_collection.delete_one({"user_name":user_name})
    return jsonify({"message": "your job application successfully deleted"})
@app.route("/view_job/<int:job_id>", methods=["GET"])
def view_job(job_id):
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("""
SELECT j.id, j.title, j.description, j.skills, j.experience, j.projects, j.education, j.qualifications, 
       j.package, j.location, c.name, c.logo_url, c.career_page_url 
FROM job_descriptions j 
JOIN companies c ON j.company_id = c.id 
WHERE j.id = ?
""", (job_id,))


    job = cursor.fetchone()
    conn.close()
    
    if job is None:
        return "Job not found", 404
    
    job_data = {
    "id": job[0],
    "title": job[1],
    "description": job[2],
    "skills": job[3],
    "experience": job[4],
    "projects": job[5],
    "education": job[6],
    "qualifications": job[7],
    "package": job[8],        # Add package
    "location": job[9],       # Add location
    "company": job[10],
    "logo_url": job[11],
    "career_page": job[12]
}
    
    return render_template("view_job.html", job=job_data)

def get_db_connection1():
    conn = sqlite3.connect("jobs.db")
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

# Fetch all jobs with company details
def fetch_jobs():
    conn = get_db_connection1()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT j.id, j.title, j.description, j.skills, j.experience, j.projects, j.education, j.qualifications, c.name AS company_name
        FROM job_descriptions j
        JOIN companies c ON j.company_id = c.id
    """)
    jobs = cursor.fetchall()
    conn.close()
    return jobs

# Fetch a single job by ID
def fetch_job_by_id(job_id):
    conn = get_db_connection1()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT j.id, j.title, j.description, j.skills, j.experience, j.projects, j.education, j.qualifications, j.company_id
        FROM job_descriptions j
        WHERE j.id = ?
    """, (job_id,))
    job = cursor.fetchone()
    conn.close()
    return job

# Fetch all companies with full details
def fetch_companies():
    conn = get_db_connection1()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, logo_url, career_page_url FROM companies")
    companies = cursor.fetchall()
    conn.close()
    return companies

# Insert a new job
def insert_job(title, description, skills, experience, projects, education, qualifications, company_id):
    conn = get_db_connection1()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO job_descriptions (title, description, skills, experience, projects, education, qualifications, company_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (title, description, skills, experience, projects, education, qualifications, company_id))
    conn.commit()
    conn.close()

# Update an existing job
def update_job(job_id, title, description, skills, experience, projects, education, qualifications, company_id):
    conn = get_db_connection1()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE job_descriptions
        SET title = ?, description = ?, skills = ?, experience = ?, projects = ?, education = ?, qualifications = ?, company_id = ?
        WHERE id = ?
    """, (title, description, skills, experience, projects, education, qualifications, company_id, job_id))
    conn.commit()
    conn.close()

# Delete a job
def delete_job(job_id):
    conn = get_db_connection1()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM job_descriptions WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()

# Insert a new company
def insert_company(name, logo_url, career_page_url):
    conn = get_db_connection1()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO companies (name, logo_url, career_page_url)
            VALUES (?, ?, ?)
        """, (name, logo_url, career_page_url))
        conn.commit()
        flash("Company added successfully!", "success")
    except sqlite3.IntegrityError:
        flash("Company name already exists!", "error")
    finally:
        conn.close()

# Update an existing company
def update_company(company_id, name, logo_url, career_page_url):
    conn = get_db_connection1()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE companies
            SET name = ?, logo_url = ?, career_page_url = ?
            WHERE id = ?
        """, (name, logo_url, career_page_url, company_id))
        conn.commit()
        flash("Company updated successfully!", "success")
    except sqlite3.IntegrityError:
        flash("Company name already exists!", "error")
    finally:
        conn.close()

# Route to manage jobs (display and handle form submission)
@app.route("/admin_jobs", methods=["GET", "POST"])
def admin_jobs():
    # Check if the user is logged in as an admin
    if 'admin_name' not in session or session.get('admin_role') != 'admin':
        return jsonify({"error": "Admin not logged in"}), 401

    if request.method == "POST":
        # Extract form data
        title = request.form.get("title")
        description = request.form.get("description")
        skills = request.form.get("skills")
        experience = request.form.get("experience")
        projects = request.form.get("projects")
        education = request.form.get("education")
        qualifications = request.form.get("qualifications")
        company_id = request.form.get("company_id")
        job_id = request.form.get("job_id")

        if job_id:  # Update existing job
            update_job(job_id, title, description, skills, experience, projects, education, qualifications, company_id)
            flash("Job updated successfully!", "success")
        else:  # Insert new job
            insert_job(title, description, skills, experience, projects, education, qualifications, company_id)
            flash("Job added successfully!", "success")

        return redirect(url_for("admin_jobs"))

    # Fetch all jobs and companies for display
    jobs = fetch_jobs()
    companies = fetch_companies()
    return render_template("admin_jobs.html", jobs=jobs, companies=companies)

# Route to add a new company
@app.route("/add_company", methods=["POST"])
def add_company():
    # Check if the user is logged in as an admin
    if 'admin_name' not in session or session.get('admin_role') != 'admin':
        return jsonify({"error": "Admin not logged in"}), 401

    # Extract form data
    company_name = request.form.get("company_name")
    logo_url = request.form.get("logo_url")
    career_page_url = request.form.get("career_page_url")

    # Insert the new company
    insert_company(company_name, logo_url, career_page_url)
    return redirect(url_for("admin_jobs"))

# Route to update a company
@app.route("/update_company/<int:company_id>", methods=["POST"])
def update_company_route(company_id):
    # Check if the user is logged in as an admin
    if 'admin_name' not in session or session.get('admin_role') != 'admin':
        return jsonify({"error": "Admin not logged in"}), 401

    # Extract form data
    company_name = request.form.get("company_name")
    logo_url = request.form.get("logo_url")
    career_page_url = request.form.get("career_page_url")

    # Update the company
    update_company(company_id, company_name, logo_url, career_page_url)
    return redirect(url_for("admin_jobs"))

# Route to fetch a single job by ID (for editing)
@app.route("/admin/jobs/<int:job_id>", methods=["GET"])
def get_job(job_id):
    job = fetch_job_by_id(job_id)
    if job:
        return jsonify({
            "id": job["id"],
            "title": job["title"],
            "description": job["description"],
            "skills": job["skills"],
            "experience": job["experience"],
            "projects": job["projects"],
            "education": job["education"],
            "qualifications": job["qualifications"],
            "company_id": job["company_id"]
        })
    return jsonify({"error": "Job not found"}), 404

# Route to delete a job
@app.route("/delete_jobs/<int:job_id>", methods=["POST"])
def delete_jobs(job_id):
    delete_job(job_id)
    flash("Job deleted successfully!", "success")
    return redirect(url_for("admin_jobs"))
@app.route("/suggestions")
def get_suggestions():
    query = request.args.get("query", "").lower()

    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch suggestions from all fields
    cursor.execute("""
        SELECT DISTINCT title FROM job_descriptions WHERE LOWER(title) LIKE ?
        UNION
        SELECT DISTINCT location FROM job_descriptions WHERE LOWER(location) LIKE ?
        UNION
        SELECT DISTINCT name FROM companies WHERE LOWER(name) LIKE ?
    """, (f"%{query}%", f"%{query}%", f"%{query}%"))

    suggestions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify(suggestions)
@app.route("/get_job_id")
def get_job_id():
    value = request.args.get("value")  # Get the search value

    conn = get_db_connection()
    cursor = conn.cursor()

    # Search for job ID across title, location, and company fields
    cursor.execute("""
        SELECT j.id 
        FROM job_descriptions j 
        JOIN companies c ON j.company_id = c.id 
        WHERE j.title = ? OR j.location = ? OR c.name = ?
    """, (value, value, value))

    job_id = cursor.fetchone()
    conn.close()
    return jsonify({"job_id": job_id[0] if job_id else None})
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
import google.generativeai as genai
genai.configure(api_key=google_api_key)


model = genai.GenerativeModel('gemini-1.5-pro')  # Use the correct model name

# Chatbot route
@app.route("/chatbot", methods=["GET", "POST"])
def chatbot_interaction():
    if request.method == "POST":
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        # Predefined responses for application-related queries
        if "resume" in user_input.lower():
            chatbot_response = "You can upload your resume by visiting the 'Resume' section. Here's the link: <a href='/resume'>Upload Resume</a>."
        elif "apply" in user_input.lower() or "job" in user_input.lower():
            chatbot_response = "You can apply for jobs by visiting the 'Apply Job' section. Here's the link: <a href='/apply_job_user'>Apply for Jobs</a>."
        elif "status" in user_input.lower() or "application" in user_input.lower():
            chatbot_response = "You can check your application status by visiting the 'Success' section. Here's the link: <a href='/success'>Check Application Status</a>."
        elif "admin" in user_input.lower():
            chatbot_response = "Admin-related queries can be addressed by logging in as an admin. Here's the link: <a href='/admin_login'>Admin Login</a>."
        elif "help" in user_input.lower():
            chatbot_response = "Here are some helpful links:<br>"
            chatbot_response += "<a href='/resume'>Upload Resume</a><br>"
            chatbot_response += "<a href='/apply_job_user'>Apply for Jobs</a><br>"
            chatbot_response += "<a href='/success'>Check Application Status</a><br>"
            chatbot_response += "<a href='/admin_login'>Admin Login</a><br>"
        else:
            # Use Google Gemini for general queries
            try:
                response = model.generate_content(
                    f"You are a helpful assistant that provides information about job applications, resumes, and career advice. If the question is unrelated to these topics, politely inform the user. User: {user_input}"
                )
                chatbot_response = response.text

                # If the response indicates the question is unrelated, provide a polite message
                if "unrelated" in chatbot_response.lower() or "not sure" in chatbot_response.lower():
                    chatbot_response = "I'm here to help with job applications, resumes, and career advice. If you have questions outside these topics, please contact support or visit our help center."

            except Exception as e:
                print(f"Error calling Google Gemini API: {e}")
                chatbot_response = "Sorry, I'm unable to process your request at the moment. Please try again later."

        return jsonify({"response": chatbot_response})

    return render_template("chatbot.html")
def get_db_connection():
    conn = sqlite3.connect('jobs_database.db')
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

# Route to render the job listings page
@app.route('/api_job')
def api_job():
    conn = get_db_connection()
    jobs = conn.execute('SELECT * FROM jobs').fetchall()  # Fetch all jobs from the database
    conn.close()
    return render_template('api_job.html', jobs=jobs)

# Route to fetch suggestions for the search bar
@app.route('/api_suggestions')
def api_suggestions():
    query = request.args.get('query', '').lower()
    conn = get_db_connection()
    jobs = conn.execute('SELECT * FROM jobs').fetchall()
    conn.close()

    
    suggestions = set()
    for job in jobs:
        if query in job['job_title'].lower():
            suggestions.add(job['job_title'])
        if query in job['location'].lower():
            suggestions.add(job['location'])
        if query in job['company_name'].lower():
            suggestions.add(job['company_name'])

    return jsonify(list(suggestions))

# Route to view a specific job
@app.route('/api_view_job/<int:job_id>')
def api_view_job(job_id):
    conn = get_db_connection()
    job = conn.execute('SELECT * FROM jobs WHERE id = ?', (job_id,)).fetchone()
    conn.close()
    return render_template("api_view_job.html",job=job)

# Route to apply for a job

API_URL = "https://jobs-api14.p.rapidapi.com/v2/list"
HEADERS = {
    "x-rapidapi-key": "26a59f0792msh168576abcd7d7c6p10fb95jsndf4892095503",
    "x-rapidapi-host": "jobs-api14.p.rapidapi.com"
}

@app.route('/api_fetch_store_jobs', methods=['GET'])
def fetch_store_jobs():
    search_query = request.args.get('query', '')

    if not search_query:
        return jsonify({"error": "No search query provided"}), 400

    # Fetch jobs from API based on search query
    query_params = {"query": search_query, "location": "india"}  
    response = requests.get(API_URL, headers=HEADERS, params=query_params)

    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch jobs"}), 500

    jobs_data = response.json().get("jobs", [])  

    conn = sqlite3.connect("jobs_database.db")
    cursor = conn.cursor()

    # Create jobs table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            location TEXT,
            company_name TEXT,
            company_logo TEXT,
            url TEXT,
            description TEXT,
            employment_type TEXT,
            date_posted TEXT
        )
    ''')

    # Insert fetched jobs into the database
    for job in jobs_data:
        cursor.execute('''
            INSERT INTO jobs (title, location, company_name, company_logo, url, description, employment_type, date_posted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job.get("title", "N/A"),
            job.get("location", "N/A"),
            job.get("company", "N/A"),
            job.get("image", "N/A"),  
            job.get("jobProviders", [{}])[0].get("url", "N/A"),  
            job.get("description", "N/A"),
            job.get("employmentType", "N/A"),
            job.get("datePosted", "N/A")
        ))

    conn.commit()
    
    # Fetch stored jobs
    cursor.execute("SELECT id, title, location, company_name FROM jobs WHERE LOWER(title) LIKE ? OR LOWER(location) LIKE ? OR LOWER(company_name) LIKE ?", 
                   (f"%{search_query}%", f"%{search_query}%", f"%{search_query}%"))

    stored_jobs = [{"id": job[0], "title": job[1], "location": job[2], "company_name": job[3]} for job in cursor.fetchall()]
    
    conn.close()

    return jsonify(stored_jobs)
@app.route("/api_apply_job/<int:job_id>", methods=["POST"])
def api_apply_job(job_id):
    # Decode the job_title to handle spaces and special characters
    print(f"Job title received: {job_id}")

    if 'user_name' not in session or session.get('user_role') != 'user':
        return jsonify({"error": "User not logged in"}), 401

    user_name = session['user_name']
    user = collection.find_one({"user_name": user_name})

    if not user:
        return jsonify({"error": "User not found"}), 404

    # Fetch the job details from the database using job_title
    conn = sqlite3.connect("jobs_database.db")
    cursor = conn.cursor()
    cursor.execute('SELECT job_title, company_name, company_logo FROM jobs')
    jobs = cursor.fetchone()
    conn.close()

    if not jobs:
        return jsonify({"error": "Job not found"}), 404

    job_title, company_name, logo_url = jobs  # Extract job title and company name

    # Check if the user has already applied for this job
    existing_application = job_applications_collection.find_one({
        "user_name": user_name,
        "job_title": job_title
    })

    if existing_application:
        return jsonify({"error": "You have already applied for this job"}), 400

    # Use the user's existing resume or generate a new one
    if user.get('resume_pdf_id'):
        resume_pdf_id = user['resume_pdf_id']
        resume_text = user.get('upload_text', '')
    else:
        return jsonify({"error": "No resume found. Please upload resume first."}), 400

    # Save the job application
    application_data = {
        "user_name": user_name,
        "job_title": job_title,
        "company_name": company_name,
        "logo_url":logo_url,  # Add company name to the application data
        "resume_pdf_id": resume_pdf_id,
        "resume_text": resume_text,
        "ats_score": user["ats_score"],
        "missing_skills": user["missing_skills"],
        "status": "pending",  # Initial status
        "application_date": ist_time_str
    }
    job_applications_collection.insert_one(application_data)
    collection.update_one(
        {"user_name": user_name},
        {"$set": {"application_status": list(application_data)}}
    )

    # Send confirmation email with job title and company name
    try:
        send_confirmation_email(user['email'], job_title, company_name,logo_url)
    except Exception as e:
        print(f"Failed to send confirmation email: {e}")

    return jsonify({
        "message": "Job application submitted successfully!",
        "ats_score": user["ats_score"],
        "missing_skills": user["missing_skills"],
        "status": "pending"
    })
if __name__ == "__main__":
    app.run(debug=os.getenv('DEBUG', 'False') == 'True')
