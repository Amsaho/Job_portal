import requests
import sqlite3

# Step 1: Fetch data from the API
url = "https://linkedin-job-api.p.rapidapi.com/job/search"
querystring = {"keyword": "Software Engineer","page": "1"}
headers = {
    "x-rapidapi-key": "26a59f0792msh168576abcd7d7c6p10fb95jsndf4892095503",
    "x-rapidapi-host": "linkedin-job-api.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)
data = response.json()

# Step 2: Create a SQLite database and table
conn = sqlite3.connect("jobs_database.db")  # Create or connect to the database
cursor = conn.cursor()

# Create a table to store job details
cursor.execute("""
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_logo TEXT,
    job_title TEXT,
    salary TEXT,
    ago_time TEXT,
    description TEXT,
    company_name TEXT,
    location TEXT,
    url TEXT
)
""")
conn.commit()

# Step 3: Insert data into the database
for job in data["data"]:
    company_logo = job["companyDetails"]["companyLogo"]
    job_title = job["title"]
    salary = job.get("salary", "Not specified")  # Use .get() to handle missing keys
    ago_time = job["agoTime"]
    description = job["description"]
    company_name = job["companyDetails"]["name"]
    location = job["companyDetails"]["location"]
    url=job["jobPostingUrl"]

    # Insert the job details into the database
    cursor.execute("""
    INSERT INTO jobs (company_logo, job_title, salary, ago_time, description, company_name, location , url)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (company_logo, job_title, salary, ago_time, description, company_name, location,url))

conn.commit()  # Save changes to the database
conn.close()  # Close the connection

print("Data has been successfully stored in the database.")