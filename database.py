import sqlite3

# Function to create the database and tables
def create_database():
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()

    # Create companies table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS companies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        logo_url TEXT,
        career_page_url TEXT
    )
    """)

    # Create job_descriptions table with company_id as a foreign key
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_descriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        skills TEXT,
        experience TEXT,
        projects TEXT,
        education TEXT,
        qualifications TEXT,
        package TEXT,  -- New field for salary/compensation
        location TEXT,  -- New field for job location
        company_id INTEGER,
        FOREIGN KEY (company_id) REFERENCES companies (id)
    )
    """)

    conn.commit()
    conn.close()

# Function to insert a company into the companies table
def insert_company(name, logo_url, career_page_url):
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()

    # Check if the company already exists
    cursor.execute("SELECT id FROM companies WHERE name = ?", (name,))
    existing_company = cursor.fetchone()

    if existing_company:
        print(f"Company '{name}' already exists. Skipping insertion.")
    else:
        # Insert the new company
        cursor.execute("""
        INSERT INTO companies (name, logo_url, career_page_url)
        VALUES (?, ?, ?)""", (name, logo_url, career_page_url))
        print(f"Company '{name}' added successfully.")

    conn.commit()
    conn.close()

# Function to insert a job into the job_descriptions table
def insert_job(title, description, skills, experience, projects, education, qualifications, package, location, company_id):
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO job_descriptions (title, description, skills, experience, projects, education, qualifications, package, location, company_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
    (title, description, skills, experience, projects, education, qualifications, package, location, company_id))
    conn.commit()
    conn.close()

# Function to fetch all jobs with company details
def fetch_jobs():
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("""
    SELECT j.title, j.description, j.skills, j.experience, j.projects, j.education, j.qualifications, j.package, j.location, c.name, c.logo_url, c.career_page_url
    FROM job_descriptions j
    JOIN companies c ON j.company_id = c.id
    """)
    jobs = cursor.fetchall()
    conn.close()
    return jobs

# Main function to set up the database and insert sample data
if __name__ == "__main__":
    # Create the database and tables
    create_database()

    # Insert 20 companies
    companies = [
        # Product-Based Companies
        ("Google", "https://logo.clearbit.com/google.com", "https://careers.google.com"),
        ("Amazon", "https://logo.clearbit.com/amazon.com", "https://www.amazon.jobs"),
        ("Microsoft", "https://logo.clearbit.com/microsoft.com", "https://careers.microsoft.com"),
        ("Apple", "https://logo.clearbit.com/apple.com", "https://www.apple.com/careers"),
        ("Meta", "https://logo.clearbit.com/meta.com", "https://www.metacareers.com"),
        ("Tesla", "https://logo.clearbit.com/tesla.com", "https://www.tesla.com/careers"),
        ("Netflix", "https://logo.clearbit.com/netflix.com", "https://jobs.netflix.com"),
        ("IBM", "https://logo.clearbit.com/ibm.com", "https://www.ibm.com/careers"),
        ("Intel", "https://logo.clearbit.com/intel.com", "https://www.intel.com/content/www/us/en/jobs/jobs-at-intel.html"),
        ("Oracle", "https://logo.clearbit.com/oracle.com", "https://www.oracle.com/careers"),

        # Service-Based Companies
        ("TCS", "https://logo.clearbit.com/tcs.com", "https://www.tcs.com/careers"),
        ("Infosys", "https://logo.clearbit.com/infosys.com", "https://www.infosys.com/careers"),
        ("Wipro", "https://logo.clearbit.com/wipro.com", "https://careers.wipro.com"),
        ("Accenture", "https://logo.clearbit.com/accenture.com", "https://www.accenture.com/in-en/careers"),
        ("Cognizant", "https://logo.clearbit.com/cognizant.com", "https://careers.cognizant.com"),
        ("HCL Technologies", "https://logo.clearbit.com/hcltech.com", "https://www.hcltech.com/careers"),
        ("Capgemini", "https://logo.clearbit.com/capgemini.com", "https://www.capgemini.com/careers"),
        ("Tech Mahindra", "https://logo.clearbit.com/techmahindra.com", "https://www.techmahindra.com/careers"),
        ("Deloitte", "https://logo.clearbit.com/deloitte.com", "https://www2.deloitte.com/global/en/careers.html"),
        ("EY", "https://logo.clearbit.com/ey.com", "https://careers.ey.com")
    ]

    for company in companies:
        insert_company(*company)

    # Fetch company IDs for job insertion
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM companies")
    company_ids = {name: id for id, name in cursor.fetchall()}
    conn.close()

    # Insert sample job descriptions with company IDs, package, and location
    job_list = [
        # Product-Based Jobs
        ("Software Engineer", 
         "Develop, test, and maintain software applications using Python and JavaScript.",
         "Python, JavaScript, React, SQL",
         "2+ years in software development",
         "Built an e-commerce platform, developed REST APIs",
         "Bachelor’s in Computer Science",
         "Certified in Full-Stack Development",
         "$120,000 - $150,000",  # Package
         "Mountain View, CA",    # Location
         company_ids["Google"]),
        
        ("Data Scientist",
         "Analyze data, build predictive models, and use machine learning techniques.",
         "Python, Machine Learning, SQL, Data Visualization",
         "3+ years in data science",
         "Developed fraud detection system, predictive analytics for sales forecasting",
         "Master’s in Data Science",
         "Google Data Analytics Certification",
         "$130,000 - $160,000",  # Package
         "Seattle, WA",           # Location
         company_ids["Amazon"]),

        ("Project Manager",
         "Manage project timelines, coordinate teams, and ensure timely delivery of products.",
         "Project Management, Agile, Scrum, Leadership",
         "5+ years in project management",
         "Managed software development projects, cloud migration projects",
         "MBA in Project Management",
         "PMP Certification",
         "$110,000 - $140,000",  # Package
         "Redmond, WA",          # Location
         company_ids["Microsoft"]),

        ("Cybersecurity Analyst",
         "Monitor and secure IT systems, prevent cyber threats, and perform penetration testing.",
         "Cybersecurity, Ethical Hacking, SIEM, Firewalls",
         "3+ years in cybersecurity",
         "Implemented security measures for a financial institution",
         "Bachelor’s in Cybersecurity or related field",
         "Certified Ethical Hacker (CEH)",
         "$100,000 - $130,000",  # Package
         "Cupertino, CA",        # Location
         company_ids["Apple"]),

        ("Cloud Engineer",
         "Design, deploy, and manage cloud infrastructure using AWS, Azure, or GCP.",
         "AWS, Azure, Docker, Kubernetes, Terraform",
         "3+ years in cloud computing",
         "Migrated enterprise applications to AWS cloud",
         "Bachelor’s in Computer Science",
         "AWS Solutions Architect Certification",
         "$115,000 - $145,000",  # Package
         "Menlo Park, CA",       # Location
         company_ids["Meta"]),

        # Service-Based Jobs
        ("Software Developer",
         "Develop and maintain software solutions for clients.",
         "Java, Spring Boot, Microservices, SQL",
         "2+ years in software development",
         "Developed REST APIs for banking applications",
         "Bachelor’s in Computer Science",
         "Oracle Certified Java Programmer",
         "$90,000 - $120,000",   # Package
         "Mumbai, India",        # Location
         company_ids["TCS"]),

        ("Data Analyst",
         "Analyze client data and generate insights for business decisions.",
         "Python, SQL, Tableau, Excel",
         "2+ years in data analysis",
         "Created dashboards for sales performance analysis",
         "Bachelor’s in Data Science",
         "Tableau Desktop Specialist Certification",
         "$80,000 - $110,000",   # Package
         "Bangalore, India",     # Location
         company_ids["Infosys"]),

        ("IT Consultant",
         "Provide IT solutions and consulting services to clients.",
         "IT Infrastructure, Cloud Computing, Project Management",
         "5+ years in IT consulting",
         "Implemented ERP systems for multiple clients",
         "Bachelor’s in Information Technology",
         "ITIL Certification",
         "$95,000 - $125,000",   # Package
         "New York, NY",         # Location
         company_ids["Accenture"]),

        ("Network Engineer",
         "Design and maintain network infrastructure for clients.",
         "Cisco, Firewalls, VPN, Network Security",
         "3+ years in network engineering",
         "Designed and implemented secure network solutions",
         "Bachelor’s in Network Engineering",
         "CCNA Certification",
         "$85,000 - $115,000",  # Package
         "Hyderabad, India",     # Location
         company_ids["Wipro"]),

        ("Business Analyst",
         "Analyze business processes and recommend improvements.",
         "Business Analysis, Agile, SQL, Power BI",
         "3+ years in business analysis",
         "Improved client onboarding process by 30%",
         "Bachelor’s in Business Administration",
         "CBAP Certification",
         "$90,000 - $120,000",  # Package
         "Chennai, India",       # Location
         company_ids["Cognizant"])
    ]

    for job in job_list:
        insert_job(*job)
    
    print("Database setup complete with 20 companies and job roles!")

    # Fetch and display jobs
    jobs = fetch_jobs()
    for job in jobs:
        title, description, skills, experience, projects, education, qualifications, package, location, company_name, logo_url, career_page_url = job
        print(f"Title: {title}")
        print(f"Company: {company_name}")
        print(f"Logo: {logo_url}")
        print(f"Career Page: {career_page_url}")
        print(f"Description: {description}")
        print(f"Skills: {skills}")
        print(f"Experience: {experience}")
        print(f"Projects: {projects}")
        print(f"Education: {education}")
        print(f"Qualifications: {qualifications}")
        print(f"Package: {package}")
        print(f"Location: {location}")
        print("-" * 50)