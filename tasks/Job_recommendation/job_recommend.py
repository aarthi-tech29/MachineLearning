
def job_portal():

    # ---------------- USER INPUT ----------------
    user_role = input("Enter your role: ").lower().strip()

    user_skills = input("Enter your skills (comma separated): ").lower().split(",")
    user_skills = set([s.strip() for s in user_skills])

    # ---------------- ROLE-SKILL REFERENCE ----------------
    role_skill_reference = {
        "java developer": {"java", "spring", "hibernate", "sql"},
        "python developer": {"python", "django", "flask"},
        "data analyst": {"excel", "sql", "power bi", "tableau", "data visualization"},
        "data scientist": {"python", "machine learning", "pandas", "numpy", "statistics"},
        "data engineer": {"python", "sql", "spark", "hadoop", "etl", "aws"},
        "ml engineer": {"python", "machine learning", "tensorflow", "pytorch"}
    }

    # ---------------- JOB DATABASE ----------------
    jobs = [
        {"company": "ABC Corp", "role": "java developer", "skills": {"java", "spring", "sql"}},
        {"company": "XYZ Corp", "role": "java developer", "skills": {"python", "django"}},  # wrong

        {"company": "Google", "role": "data scientist", "skills": {"python", "machine learning", "pandas"}},
        {"company": "Amazon", "role": "data engineer", "skills": {"python", "sql", "spark", "aws"}},
        {"company": "Microsoft", "role": "data analyst", "skills": {"excel", "sql", "power bi"}},
        {"company": "Meta", "role": "ml engineer", "skills": {"python", "machine learning", "tensorflow"}},

        {"company": "FakeData Inc", "role": "data scientist", "skills": {"excel", "power bi"}}  # wrong
    ]

    print("\n--- JOB RESULTS ---\n")

    primary_jobs = []
    related_jobs = []

    for job in jobs:

        job_role = job["role"]
        job_skills = set([s.lower() for s in job["skills"]])  # normalize

        # ---------------- STEP 1: VALIDATION ----------------
        reference_skills = role_skill_reference.get(job_role, set())

        if len(job_skills) == 0:
            continue

        common = job_skills.intersection(reference_skills)
        match_ratio = len(common) / len(job_skills)

        # remove wrong company data
        if match_ratio < 0.3:
            continue

        # ---------------- STEP 2: USER MATCH ----------------
        matched = user_skills.intersection(job_skills)

        if len(matched) == 0:
            continue

        score = (len(matched) / len(job_skills)) * 100

        # ---------------- MATCH LEVEL ----------------
        if score >= 70:
            level = "Strong Match"
        elif score >= 40:
            level = "Moderate Match"
        else:
            level = "Weak Match"

        job_data = {
            "company": job["company"],
            "role": job_role,
            "matched": len(matched),
            "total": len(job_skills),
            "score": score,
            "level": level
        }

        # ---------------- ROLE-FIRST LOGIC ----------------
        if job_role == user_role:
            primary_jobs.append(job_data)
        else:
            related_jobs.append(job_data)

    # ---------------- SORT ----------------
    primary_jobs.sort(key=lambda x: x["score"], reverse=True)
    related_jobs.sort(key=lambda x: x["score"], reverse=True)

    # ---------------- OUTPUT ----------------
    if not primary_jobs and not related_jobs:
        print("No jobs found")
        return

    print("PRIMARY JOBS (Role)\n")

    if primary_jobs:
        for r in primary_jobs:
            print("Company:", r["company"])
            print("Role:", r["role"])
            print("Matched Skills:", r["matched"], "/", r["total"])
            print("Match %:", round(r["score"], 2))
            print("Match Level:", r["level"])
            print("---------------------")
    else:
        print("No jobs available for your role.\n")

    print("\nRELATED JOBS (Based on Skills)\n")

    for r in related_jobs:
        print("Company:", r["company"])
        print("Role:", r["role"])
        print("Matched Skills:", r["matched"], "/", r["total"])
        print("Match %:", round(r["score"], 2))
        print("Match Level:", r["level"])
        print("---------------------")


# RUN
job_portal()

# # PERFECT MATCH
# # Role: data scientist
# # Skills: python, machine learning, pandas

# # PARTIAL MATCH
# # Role: data scientist
# # Skills: python, pandas

# # WRONG COMPANY DATA (AUTO REMOVED)
# # Role: java developer
# # Skills: python, django

# # NO PRIMARY MATCH (ONLY RELATED)
# # Role: data scientist
# # Skills: sql

# # JAVA USER (GOOD MATCH)
# # Role: java developer
# # Skills: java, sql

# # WEAK JAVA PROFILE
# # Role: java developer
# # Skills: java

# # MIXED SKILLS (CROSS DOMAIN)
# # Role: java developer
# # Skills: java, python, django

# # NO MATCH AT ALL
# # Role: data scientist
# # Skills: html, css

# # Primary jobs first
# # Related jobs next
# # Skill-based ranking
# # Wrong company data removed
# # Partial matching allowed

