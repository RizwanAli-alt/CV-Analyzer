"""
Mock Job Data for Testing & Demo Mode.

Provides realistic job listings for all three portals when live
scraping is unavailable (network restricted environment, CI/CD, demos).

Mock data covers a wide range of tech roles, companies, locations,
and modalities — filtered by query to simulate real search results.
"""

import re
import random
from datetime import datetime, timedelta
from typing import List
from .base import JobListing

_MOCK_JOBS = [
    # ── Python / Backend ───────────────────────────────────────────────────
    dict(title="Python Backend Developer", company="Systems Limited", location="Lahore, Pakistan",
         modality="Hybrid", portal="LinkedIn", url="https://www.linkedin.com/jobs/view/python-backend-001",
         description="We are looking for a Python Backend Developer with 2+ years of experience in Django and FastAPI. "
                     "You will design RESTful APIs, work with PostgreSQL databases, and deploy on AWS. "
                     "Strong knowledge of Git, Docker, and CI/CD pipelines required. "
                     "Skills: Python, Django, FastAPI, PostgreSQL, Docker, AWS, Git, REST API"),

    dict(title="Senior Python Engineer", company="TRG Pakistan", location="Islamabad, Pakistan",
         modality="On-site", portal="Rozee.pk", url="https://www.rozee.pk/job/senior-python-engineer",
         description="Senior Python Engineer to lead our backend team. "
                     "Must have expertise in Python, Django, Celery, Redis, and PostgreSQL. "
                     "Experience with microservices architecture and Docker/Kubernetes is a plus. "
                     "Skills: Python, Django, Celery, Redis, PostgreSQL, Microservices, Docker, Kubernetes"),

    dict(title="Python Developer (Remote)", company="Upwork Client – FinTech", location="Remote",
         modality="Remote", portal="Indeed", url="https://www.indeed.com/viewjob?jk=python-remote-001",
         description="Looking for a Python Developer to build financial data pipelines. "
                     "Experience with Pandas, NumPy, and SQL required. FastAPI or Flask for API development. "
                     "Machine Learning knowledge is a strong advantage. "
                     "Skills: Python, Pandas, NumPy, SQL, FastAPI, Flask, Machine Learning"),

    dict(title="Django REST Framework Developer", company="Arbisoft", location="Lahore, Pakistan",
         modality="Hybrid", portal="LinkedIn", url="https://www.linkedin.com/jobs/view/django-dev-arbisoft",
         description="Join our product team to build scalable Django applications. "
                     "Must know Django REST Framework, PostgreSQL, Celery, Redis. "
                     "Experience with React.js for front-end integration preferred. "
                     "Skills: Django, REST API, PostgreSQL, Celery, Redis, React, Python"),

    # ── Data Science / ML ──────────────────────────────────────────────────
    dict(title="Data Scientist", company="Jazz Pakistan", location="Islamabad, Pakistan",
         modality="Hybrid", portal="LinkedIn", url="https://www.linkedin.com/jobs/view/data-scientist-jazz",
         description="Data Scientist to analyse telecom data and build predictive models. "
                     "Proficiency in Python, Scikit-learn, TensorFlow, Pandas required. "
                     "SQL experience for data extraction. Strong communication skills. "
                     "Skills: Python, Scikit-learn, TensorFlow, Pandas, SQL, Machine Learning, Data Science, Communication"),

    dict(title="Machine Learning Engineer", company="Afiniti", location="Islamabad, Pakistan",
         modality="On-site", portal="Rozee.pk", url="https://www.rozee.pk/job/ml-engineer-afiniti",
         description="Build and deploy ML models for real-time call centre optimization. "
                     "Expert-level Python, PyTorch, and TensorFlow required. "
                     "Experience with Docker, Kubernetes, and AWS SageMaker. "
                     "Skills: Python, PyTorch, TensorFlow, Machine Learning, Deep Learning, Docker, Kubernetes, AWS"),

    dict(title="Junior Data Analyst", company="PTCL", location="Islamabad, Pakistan",
         modality="On-site", portal="Indeed", url="https://www.indeed.com/viewjob?jk=data-analyst-ptcl",
         description="Entry-level data analyst position. "
                     "SQL proficiency required, Python/Pandas a plus. Excel and Power BI for reporting. "
                     "Strong analytical and communication skills. "
                     "Skills: SQL, Python, Pandas, Excel, Power BI, Analysis, Communication"),

    dict(title="AI/ML Research Engineer", company="10Pearls", location="Karachi, Pakistan",
         modality="Hybrid", portal="LinkedIn", url="https://www.linkedin.com/jobs/view/ai-ml-10pearls",
         description="Research and implement state-of-the-art ML algorithms. "
                     "Deep expertise in PyTorch, Python, and NLP required. "
                     "Published research is a plus. Generative AI and LLM experience strongly preferred. "
                     "Skills: Python, PyTorch, Machine Learning, Deep Learning, NLP, Generative AI"),

    # ── Frontend / Full-Stack ──────────────────────────────────────────────
    dict(title="React.js Developer", company="Techlogix", location="Lahore, Pakistan",
         modality="Hybrid", portal="Rozee.pk", url="https://www.rozee.pk/job/react-developer-techlogix",
         description="Build modern web applications using React.js and TypeScript. "
                     "Experience with Node.js, REST APIs, and MongoDB required. "
                     "Git workflow and Agile methodology knowledge. "
                     "Skills: React, TypeScript, JavaScript, Node.js, REST API, MongoDB, Git, Agile"),

    dict(title="Full Stack Developer (Django + React)", company="Confiz", location="Lahore, Pakistan",
         modality="On-site", portal="LinkedIn", url="https://www.linkedin.com/jobs/view/fullstack-confiz",
         description="Full stack role: Django backend + React frontend. "
                     "Must be comfortable with PostgreSQL, REST APIs, and CI/CD pipelines. "
                     "Strong problem-solving and teamwork skills required. "
                     "Skills: Django, React, Python, JavaScript, PostgreSQL, REST API, CI/CD, Git, Teamwork"),

    dict(title="Vue.js Frontend Developer", company="Inbox Business Technologies", location="Karachi, Pakistan",
         modality="Remote", portal="Indeed", url="https://www.indeed.com/viewjob?jk=vuejs-frontend",
         description="Develop responsive SPA applications with Vue.js 3 and TypeScript. "
                     "RESTful API integration. Experience with Webpack and CSS/SASS. "
                     "Skills: Vue.js, TypeScript, JavaScript, HTML, CSS, SASS, REST API, Webpack"),

    # ── DevOps / Cloud ─────────────────────────────────────────────────────
    dict(title="DevOps Engineer", company="Netsol Technologies", location="Lahore, Pakistan",
         modality="Hybrid", portal="LinkedIn", url="https://www.linkedin.com/jobs/view/devops-netsol",
         description="Manage CI/CD pipelines, Kubernetes clusters, and AWS infrastructure. "
                     "Expert in Docker, Terraform, Ansible, Jenkins, and Linux. "
                     "Monitor system health and implement observability. "
                     "Skills: DevOps, Docker, Kubernetes, AWS, Terraform, Ansible, Jenkins, Linux, CI/CD"),

    dict(title="Cloud Engineer – AWS", company="Contour Software", location="Lahore, Pakistan",
         modality="On-site", portal="Rozee.pk", url="https://www.rozee.pk/job/aws-cloud-engineer",
         description="Design and maintain cloud infrastructure on AWS. "
                     "Strong knowledge of EC2, S3, RDS, Lambda, CloudFormation required. "
                     "Terraform for IaC. Docker and Kubernetes for containerisation. "
                     "Skills: AWS, Cloud Computing, Terraform, Docker, Kubernetes, Linux, CI/CD"),

    dict(title="Site Reliability Engineer (SRE)", company="Genetech", location="Karachi, Pakistan",
         modality="Remote", portal="Indeed", url="https://www.indeed.com/viewjob?jk=sre-genetech",
         description="SRE role to maintain 99.9% uptime of production services. "
                     "Python scripting for automation, Kubernetes orchestration, "
                     "Prometheus/Grafana for monitoring. Strong Linux background. "
                     "Skills: Python, Kubernetes, Docker, Linux, AWS, CI/CD, DevOps"),

    # ── Software Engineering (General) ─────────────────────────────────────
    dict(title="Software Engineer – Java/Spring Boot", company="i2c Inc.", location="Lahore, Pakistan",
         modality="On-site", portal="LinkedIn", url="https://www.linkedin.com/jobs/view/java-i2c",
         description="Develop microservices with Java Spring Boot and REST APIs. "
                     "Proficiency in SQL, Git, and Agile development required. "
                     "Docker and Kubernetes for deployment. Leadership potential valued. "
                     "Skills: Java, Spring Boot, REST API, SQL, Docker, Microservices, Git, Agile, Leadership"),

    dict(title="Junior Software Engineer (Fresh Graduate)", company="Folio3", location="Karachi, Pakistan",
         modality="On-site", portal="Indeed", url="https://www.indeed.com/viewjob?jk=junior-swe-folio3",
         description="Entry-level position for fresh graduates. "
                     "Training provided in Python or JavaScript. "
                     "Strong problem-solving, teamwork, and communication skills required. "
                     "Skills: Python, JavaScript, Problem-solving, Teamwork, Communication, Git"),

    dict(title="Mobile App Developer – Flutter", company="Venture Dive", location="Islamabad, Pakistan",
         modality="Hybrid", portal="Rozee.pk", url="https://www.rozee.pk/job/flutter-developer",
         description="Build cross-platform mobile apps with Flutter and Dart. "
                     "Experience integrating REST APIs and Firebase. "
                     "Agile workflow, Git. iOS/Android deployment experience. "
                     "Skills: Flutter, Dart, Mobile Development, REST API, Firebase, Git, Agile"),

    # ── Cybersecurity / QA ─────────────────────────────────────────────────
    dict(title="QA Automation Engineer", company="Motive (formerly KeepTruckin)", location="Lahore, Pakistan",
         modality="Remote", portal="LinkedIn", url="https://www.linkedin.com/jobs/view/qa-motive",
         description="Automate testing using Selenium, Pytest, and Postman. "
                     "Python scripting for test frameworks. CI/CD integration with Jenkins. "
                     "Strong attention to detail and problem-solving skills. "
                     "Skills: Python, Selenium, Pytest, Postman, CI/CD, Jenkins, Testing"),

    dict(title="Cybersecurity Analyst", company="NetSecurity", location="Islamabad, Pakistan",
         modality="On-site", portal="Indeed", url="https://www.indeed.com/viewjob?jk=cybersec-analyst",
         description="Monitor and respond to security incidents. "
                     "OWASP, JWT, OAuth, SSL/TLS knowledge required. "
                     "Web Security audit experience. Linux and scripting skills. "
                     "Skills: Cybersecurity, OWASP, Linux, Python, Web Security, SSL/TLS, JWT"),

    dict(title="Database Administrator – PostgreSQL", company="Pakistan Telecom", location="Islamabad, Pakistan",
         modality="On-site", portal="Rozee.pk", url="https://www.rozee.pk/job/dba-postgresql",
         description="Manage and optimise PostgreSQL databases. "
                     "Query optimisation, backup/recovery, replication. "
                     "Experience with MongoDB and Redis a plus. Linux server administration. "
                     "Skills: PostgreSQL, SQL, MongoDB, Redis, Linux, Database Design, NoSQL"),
]


def get_mock_listings(query: str, location: str = "", portals: List[str] = None,
                      max_results: int = 20) -> List[JobListing]:
    """
    Return mock job listings filtered by query, location, and portals.
    Simulates realistic search results for testing.
    """
    if portals is None:
        portals = ["LinkedIn", "Indeed", "Rozee.pk"]

    query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    location_lower = location.lower().strip()

    def relevance(job: dict) -> float:
        text = f"{job['title']} {job['description']}".lower()
        job_words = set(re.findall(r"\b\w{3,}\b", text))
        overlap = len(query_words & job_words) if query_words else 0.5
        loc_match = 1.5 if (not location_lower or location_lower in job["location"].lower()) else 1.0
        return overlap * loc_match

    # Filter by portals, score by relevance
    candidates = [j for j in _MOCK_JOBS if j["portal"] in portals]
    scored = sorted(candidates, key=relevance, reverse=True)

    # Add slight date variation so they look like real scraped data
    listings = []
    for i, job in enumerate(scored[:max_results]):
        age_days = random.randint(0, 25)  # all within 30-day window
        scraped = datetime.utcnow() - timedelta(days=age_days, hours=random.randint(0, 23))
        listing = JobListing(
            title=job["title"],
            company=job["company"],
            location=job["location"],
            modality=job["modality"],
            portal=job["portal"],
            url=job["url"],
            description=job["description"],
            skills_mentioned=_extract_skills_quick(job["description"]),
            scraped_at=scraped,
            match_score=0.0,
        )
        listings.append(listing)

    return listings


def _extract_skills_quick(description: str) -> List[str]:
    """Quick skill extraction from mock description's Skills: line."""
    m = re.search(r"Skills:\s*(.+)$", description, re.IGNORECASE | re.MULTILINE)
    if m:
        return [s.strip() for s in m.group(1).split(",") if s.strip()]
    return []