# ============================ Basic matching with wrong company data removed ============================
def job_portal():

    # ---------------- USER INPUT ----------------
    user_role = input("Enter your role: ").lower().strip()

    user_skills = input("Enter your skills (comma separated): ").lower().split(",")
    user_skills = set([s.strip() for s in user_skills])

    # ---------------- VALID ROLE SKILLS ----------------
    valid_role_skills = {
        "java developer": {"java", "spring", "hibernate", "sql"},
        "python developer": {"python", "django", "flask"},
        "data analyst": {"excel", "sql", "power bi", "tableau", "data visualization"},
        "data scientist": {"python", "machine learning", "pandas", "numpy", "statistics"},
        "data engineer": {"python", "sql", "spark", "hadoop", "etl", "aws"},
    }
    # To remove wrong company entries

    # ---------------- JOB DATABASE ----------------
    jobs = [
        {"company": "ABC Corp", "role": "java developer", "skills": {"java", "spring", "sql"}},
        {"company": "XYZ Corp", "role": "java developer", "skills": {"python", "django"}},  # wrong
        {"company": "Google", "role": "data scientist", "skills": {"python", "machine learning", "pandas"}},
        {"company": "Amazon", "role": "data engineer", "skills": {"python", "sql", "spark", "aws"}},
        {"company": "Microsoft", "role": "data analyst", "skills": {"excel", "sql", "power bi"}},
        {"company": "Meta", "role": "ml engineer", "skills": {"python", "machine learning"}}
    ]

    print("\n--- JOB RESULTS ---\n")

    primary_jobs = []
    related_jobs = []

    for job in jobs:

        job_role = job["role"]
        job_skills = job["skills"]

        # ---------------- STEP 1: VALIDATE COMPANY DATA ----------------
        valid_skills = valid_role_skills.get(job_role, set())

        if len(job_skills.intersection(valid_skills)) == 0: # intersection() - Checks if company gave correct skills for that role
            continue  #  remove wrong job

        # ---------------- SKILL MATCH ----------------
        matched = user_skills.intersection(job_skills) # Result = empty →  removed

        if len(matched) == 0:
            continue  # no relation at all

        score = (len(matched) / len(job_skills)) * 100

        job_data = {
            "company": job["company"],
            "role": job_role,
            "matched": len(matched),
            "total": len(job_skills),
            "score": score
        }

        # ---------------- STEP 2: PRIMARY vs RELATED ----------------
        if job_role == user_role:
            primary_jobs.append(job_data)
        else:
            related_jobs.append(job_data)

    # ---------------- SORT ----------------
    primary_jobs.sort(key=lambda x: x["score"], reverse=True)
    related_jobs.sort(key=lambda x: x["score"], reverse=True)

    # ---------------- OUTPUT ----------------
    if not primary_jobs and not related_jobs:
        print("No valid jobs found")
        return

    # ---------- PRIMARY ----------
    print("✔ PRIMARY JOBS (Role)\n")
    for r in primary_jobs:
        print("Company:", r["company"])
        print("Role:", r["role"])
        print("Matched Skills:", r["matched"], "/", r["total"])
        print("Match %:", round(r["score"], 2))
        print("---------------------")

    # ---------- RELATED ----------
    print("\nRELATED JOBS (Based on Skills)\n")
    for r in related_jobs:
        print("Company:", r["company"])
        print("Role:", r["role"])
        print("Matched Skills:", r["matched"], "/", r["total"])
        print("Match %:", round(r["score"], 2))
        print("---------------------")


# RUN
job_portal()

# PERFECT MATCH
# Role: data scientist
# Skills: python, machine learning, pandas

# PARTIAL MATCH
# Role: data scientist
# Skills: python, pandas

# WRONG COMPANY DATA
# XYZ Corp
# Role: java developer
# Skills: python, django

# NO PRIMARY MATCH, ONLY RELATED
# Role: data scientist
# Skills: sql

# JAVA USER
# Role: java developer
# Skills: java, sql

# Primary jobs first
# Related jobs next
# Skill-based ranking
# Wrong company data removed
# Partial matching allowed
# =================================================

# # ==========================================================================
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

# ======================== New trend skills automatically added in rulebook ==========================
import json
import os

# ---------------- LOAD RULE BOOK ----------------
def load_rulebook():
    if os.path.exists("role_skills.json"):
        with open("role_skills.json", "r") as f:
            data = json.load(f)
            return {k: set(v) for k, v in data.items()}
    return {}

# ---------------- SAVE RULE BOOK ----------------
def save_rulebook(role_skill_reference):
    data = {k: list(v) for k, v in role_skill_reference.items()}
    with open("role_skills.json", "w") as f:
        json.dump(data, f, indent=4)


def job_portal():

    # ---------------- USER INPUT ----------------
    user_role = input("Enter your role: ").lower().strip()

    user_skills = input("Enter your skills (comma separated): ").lower().split(",")
    user_skills = set([s.strip() for s in user_skills])

    # ---------------- BASE VALID SKILLS ----------------
    base_valid_skills = {
        "java developer": {"java", "spring", "hibernate", "sql"},
        "python developer": {"python", "django", "flask"},
        "data scientist": {"python", "machine learning", "pandas"},
        "data engineer": {"python", "sql", "spark", "aws"},
        "data analyst": {"excel", "sql", "power bi"},
        "ml engineer": {"python", "machine learning", "tensorflow"}
    }

    # ---------------- LOAD RULE BOOK ----------------
    role_skill_reference = load_rulebook()

    # ---------------- JOB DATABASE ----------------
    jobs = [
        {"company": "ABC Corp", "role": "java developer", "skills": {"java", "spring", "sql"}},
        {"company": "XYZ Corp", "role": "java developer", "skills": {"python", "django"}},  # wrong

        {"company": "Google", "role": "data scientist", "skills": {"python", "machine learning", "pandas"}},
        {"company": "OpenAI", "role": "data scientist", "skills": {"python", "llm", "generative ai"}},

        {"company": "Amazon", "role": "data engineer", "skills": {"python", "sql", "spark", "aws", "hadoop"}},
        {"company": "Microsoft", "role": "data analyst", "skills": {"excel", "sql", "power bi"}},
        {"company": "Meta", "role": "ml engineer", "skills": {"python", "machine learning", "tensorflow"}},

        {"company": "StartupX", "role": "data scientist", "skills": {"deep learning", "nlp"}}
    ]

    # ---------------- NORMALIZE JOB SKILLS ----------------
    for job in jobs:
        job["skills"] = set([s.lower() for s in job["skills"]])

    # ---------------- STEP 1: VALIDATE + LEARN ----------------
    print("\n--- UPDATING RULE BOOK ---")

    for job in jobs:
        role = job["role"]
        job_skills = job["skills"]

        valid_skills = base_valid_skills.get(role, set())

        if len(job_skills) == 0:
            continue

        common = job_skills.intersection(valid_skills)
        match_ratio = len(common) / len(job_skills)

        # skip wrong jobs (do not learn)
        if match_ratio < 0.3:
            continue

        # learn new skills
        if role not in role_skill_reference:
            role_skill_reference[role] = set()

        for skill in job_skills:
            if skill not in role_skill_reference[role]:
                print(f"Learned new skill for {role}: {skill}")
                role_skill_reference[role].add(skill)

    # ---------------- SAVE RULE BOOK ----------------
    save_rulebook(role_skill_reference)

    print("\n--- CURRENT RULE BOOK ---")
    for role, skills in role_skill_reference.items():
        print(role, "->", skills)

    print("\n--- JOB RESULTS ---\n")

    primary_jobs = []
    related_jobs = []

    # ---------------- STEP 2: MATCH JOBS ----------------
    for job in jobs:

        job_role = job["role"]
        job_skills = job["skills"]

        valid_skills = base_valid_skills.get(job_role, set())

        if len(job_skills) == 0:
            continue

        common = job_skills.intersection(valid_skills)
        match_ratio = len(common) / len(job_skills)

        # remove wrong jobs completely
        if match_ratio < 0.3:
            continue

        # ---------------- USER MATCH ----------------
        matched = user_skills.intersection(job_skills)

        if len(matched) == 0:
            continue

        score = (len(matched) / len(job_skills)) * 100

        job_data = {
            "company": job["company"],
            "role": job_role,
            "matched": len(matched),
            "total": len(job_skills),
            "score": score,
            "level": "Strong Match" if score >= 50 else "Weak Match"
        }

        # ---------------- ROLE-FIRST PRIORITY ----------------
        if job_role == user_role:
            primary_jobs.append(job_data)
        else:
            related_jobs.append(job_data)

    # ---------------- SORT ----------------
    primary_jobs.sort(key=lambda x: x["score"], reverse=True)
    related_jobs.sort(key=lambda x: x["score"], reverse=True)

    # ---------------- OUTPUT ----------------
    if not primary_jobs and not related_jobs:
        print(" No jobs found")
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

    print("RELATED JOBS (Based on Skills)\n")

    for r in related_jobs:
        print("Company:", r["company"])
        print("Role:", r["role"])
        print("Matched Skills:", r["matched"], "/", r["total"])
        print("Match %:", round(r["score"], 2))
        print("Match Level:", r["level"])
        print("---------------------")


# RUN
job_portal()

