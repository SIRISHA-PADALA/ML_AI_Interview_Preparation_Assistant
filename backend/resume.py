"""Resume parsing: PDF text -> cleaned text -> skills (dictionary match) -> interview topics.

Rule-based on purpose (simple, explainable, maintainable). To support a new skill, add one line to
SKILL_TABLE below. Every target is a (category, topic) pair that must exist in the question dataset;
pairs that don't exist are dropped at runtime, so nothing is ever recommended that has no questions.
"""
import re
import pdfplumber

CODES = {"AIML": "AI & ML", "DEVOPS": "DevOps Engineer", "REACT": "React Engineer", "SAP": "SAP Engineer",
         "SWB": "Software Interview Bank"}

# skill | group | aliases (comma separated) | targets (CODE:Topic ; CODE:Topic)
SKILL_TABLE = """
python|Programming Languages|python,python3|AIML:Python;SWB:General Programming
java|Programming Languages|java|SWB:General Programming;SWB:Languages and Frameworks
javascript|Programming Languages|javascript,es6|SWB:Front-end;SWB:Web Development
typescript|Programming Languages|typescript|SWB:Front-end;SWB:Languages and Frameworks
c++|Programming Languages|c++,cpp|SWB:Low-level Systems;SWB:Algorithms
c#|Programming Languages|c#|SWB:Languages and Frameworks
c|Programming Languages|c programming,c language|SWB:Low-level Systems
abap|Programming Languages|abap|SAP:ABAP
html|Programming Languages|html,html5|SWB:Front-end;SWB:Web Development
css|Programming Languages|css,css3|SWB:Front-end;SWB:Web Development
sql|Databases|sql,plsql,pl/sql|SWB:Database and SQL
mysql|Databases|mysql|SWB:Database and SQL
postgresql|Databases|postgresql,postgres|SWB:Database and SQL
sqlite|Databases|sqlite|SWB:Database and SQL
oracle|Databases|oracle db,oracle database|SWB:Database and SQL
mongodb|Databases|mongodb,mongo db|SWB:Database and SQL;SWB:Back-end
react|Web & Frameworks|react,react.js,reactjs|REACT:Hooks;REACT:State;REACT:Components;REACT:JSX;REACT:Lifecycle
redux|Web & Frameworks|redux|REACT:Redux
react router|Web & Frameworks|react router|REACT:Routing
jest|Web & Frameworks|jest,react testing library|REACT:Testing
node.js|Web & Frameworks|node.js,nodejs,node js|SWB:Back-end
express|Web & Frameworks|express,express.js|SWB:Back-end
flask|Web & Frameworks|flask|SWB:Back-end
django|Web & Frameworks|django|SWB:Back-end
rest api|Web & Frameworks|rest api,rest apis,restful|SWB:Back-end;SWB:Web Development
full stack|Web & Frameworks|full stack,full-stack,fullstack|SWB:Full-stack
aws|Cloud & DevOps|aws,amazon web services|DEVOPS:Cloud;SWB:DevOps
azure|Cloud & DevOps|azure|DEVOPS:Cloud;SWB:DevOps
gcp|Cloud & DevOps|gcp,google cloud|DEVOPS:Cloud;SWB:DevOps
docker|Cloud & DevOps|docker|DEVOPS:Docker;SWB:DevOps
kubernetes|Cloud & DevOps|kubernetes,k8s|DEVOPS:Kubernetes;SWB:DevOps
terraform|Cloud & DevOps|terraform|DEVOPS:Terraform
ci/cd|Cloud & DevOps|ci/cd,jenkins,github actions,gitlab ci|DEVOPS:CI/CD;DEVOPS:Pipelines
prometheus|Cloud & DevOps|prometheus,grafana|DEVOPS:Monitoring
devops|Cloud & DevOps|devops|SWB:DevOps;DEVOPS:Pipelines
linux|Cloud & DevOps|linux,unix|SWB:Low-level Systems;SWB:DevOps
git|Tools & Practices|git,github,gitlab,bitbucket|DEVOPS:Git;SWB:Version Control
testing|Tools & Practices|unit testing,selenium,pytest,junit,software testing|SWB:Software Testing
machine learning|AI, ML & Data|machine learning,ml|AIML:ML;AIML:ML Basics;AIML:Models;AIML:Training;SWB:Machine Learning
deep learning|AI, ML & Data|deep learning,neural networks,cnn|AIML:Deep Learning
tensorflow|AI, ML & Data|tensorflow|AIML:Deep Learning
pytorch|AI, ML & Data|pytorch|AIML:Deep Learning
keras|AI, ML & Data|keras|AIML:Deep Learning
scikit-learn|AI, ML & Data|scikit-learn,sklearn,scikit learn|AIML:Models;AIML:Training
pandas|AI, ML & Data|pandas|AIML:Preprocessing;AIML:Python
numpy|AI, ML & Data|numpy|AIML:Python;AIML:Preprocessing
nlp|AI, ML & Data|nlp,natural language processing|AIML:Concepts;SWB:Machine Learning
data science|AI, ML & Data|data science,data analysis|AIML:ML;AIML:Preprocessing
data engineering|AI, ML & Data|data engineering,etl,spark,hadoop|SWB:Data Engineering
sap|Enterprise (SAP)|sap|SAP:ABAP;SAP:HANA;SAP:Fiori;SAP:BASIS
hana|Enterprise (SAP)|hana,sap hana|SAP:HANA
fiori|Enterprise (SAP)|fiori,sapui5|SAP:Fiori
sap fi/co|Enterprise (SAP)|sap fi,sap co|SAP:FI;SAP:CO
sap mm/sd|Enterprise (SAP)|sap mm,sap sd|SAP:MM;SAP:SD
data structures|CS Fundamentals|data structures|SWB:Data Structures
algorithms|CS Fundamentals|algorithms,dsa|SWB:Algorithms;SWB:Data Structures
oop|CS Fundamentals|oop,object-oriented,object oriented|SWB:General Programming
system design|CS Fundamentals|system design|SWB:System Design;SWB:Distributed Systems
microservices|CS Fundamentals|microservices|SWB:System Design;SWB:Distributed Systems
networking|CS Fundamentals|computer networks,networking,tcp/ip|SWB:Networking
security|CS Fundamentals|cybersecurity,security,owasp|SWB:Security;DEVOPS:Security
"""


def _load():
    out = {}
    for line in SKILL_TABLE.strip().splitlines():
        skill, group, aliases, targets = line.split("|")
        out[skill] = dict(group=group, aliases=aliases.split(","),
                          targets=[(CODES[t.split(":")[0]], t.split(":", 1)[1]) for t in targets.split(";")])
    return out


SKILLS = _load()
GROUP_ORDER = ["Programming Languages", "Web & Frameworks", "Databases", "Cloud & DevOps", "AI, ML & Data",
               "Enterprise (SAP)", "Tools & Practices", "CS Fundamentals"]
HEADINGS = {"education": r"education|academic|academics|qualifications?", "experience": r"experience|employment|internships?|work history",
            "projects": r"projects?|personal projects|academic projects", "skills": r"skills|technologies|technical skills",
            "other": r"certifications?|achievements|awards|summary|objective|profile|interests|languages|publications|references"}
DEGREE = re.compile(r"b\.?tech|b\.?e\b|b\.?sc|bachelor|m\.?tech|m\.?sc|master|mba|ph\.?d|diploma|b\.?ca|m\.?ca|\bcgpa\b|\bgpa\b", re.I)


def extract_text(fileobj):
    with pdfplumber.open(fileobj) as pdf:
        text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    return text.strip()


def clean_text(text):
    """Lowercase, drop URLs/emails/odd punctuation, keep technical symbols (+ # . / -), collapse whitespace."""
    t = text.lower()
    t = re.sub(r"(https?://|www\.)\S+|\S+@\S+\.\S+", " ", t)
    t = re.sub(r"[^a-z0-9+#./\- \n]", " ", t)
    t = re.sub(r"(?<![a-z0-9])[.\-/]+|[.\-/]+(?![a-z0-9+#])", " ", t)   # stray punctuation, keeps node.js, ci/cd, c++
    return re.sub(r"\s+", " ", t).strip()


def find_skills(cleaned):
    padded = f" {cleaned} "
    found = []
    for skill, info in SKILLS.items():
        for alias in info["aliases"]:
            if re.search(r"(?<![a-z0-9+#])" + re.escape(alias) + r"(?![a-z0-9+#])", padded):
                found.append(skill)
                break
    return found


def split_sections(text):
    sections, cur = {"header": []}, "header"
    for line in text.splitlines():
        s = line.strip().strip(":").strip()
        hit = next((k for k, pat in HEADINGS.items() if re.fullmatch(pat, s, re.I)), None) if 0 < len(s) < 32 else None
        if hit:
            cur = hit
            sections.setdefault(cur, [])
        elif s:
            sections.setdefault(cur, []).append(line.strip())
    return sections


def recommend(skills, pair_counts):
    """Skills -> (category, topic) pairs that really exist in the dataset, ranked by how many skills point to them."""
    pairs = {}
    for s in skills:
        for pair in SKILLS[s]["targets"]:
            if pair_counts.get(pair):
                pairs.setdefault(pair, []).append(s)
    recs = [dict(category=c, topic=t, because=sorted(v), questions_available=pair_counts[(c, t)]) for (c, t), v in pairs.items()]
    recs.sort(key=lambda r: (-len(r["because"]), -r["questions_available"], r["topic"]))
    top = recs[:12]
    for s in skills:  # guarantee every detected skill appears behind at least one recommended topic
        if not any(s in r["because"] for r in top):
            top.append(next(r for r in recs if s in r["because"]))
    return top


def analyze(text, pair_counts, filename=""):
    skills = find_skills(clean_text(text))
    by_group = {g: [s for s in skills if SKILLS[s]["group"] == g] for g in GROUP_ORDER}
    by_group = {g: v for g, v in by_group.items() if v}
    sec = split_sections(text)
    edu = [l for l in sec.get("education", []) if DEGREE.search(l) or re.search(r"university|college|institute|school", l, re.I)]
    if not edu:
        edu = [l for l in text.splitlines() if DEGREE.search(l)][:4]
    years = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", text, re.I)
    recs = recommend(skills, pair_counts)
    return dict(filename=filename, skills=skills, skill_count=len(skills), skills_by_group=by_group,
                education=edu[:6], experience=sec.get("experience", [])[:12],
                experience_years=max((float(y) for y in years), default=None), projects=sec.get("projects", [])[:14],
                recommended_topics=recs,
                how=("The PDF text was extracted, lowercased and cleaned (URLs and stray punctuation removed, technical terms like "
                     "C++, Node.js and CI/CD kept). Each word or phrase was matched against a skill dictionary. Every detected skill "
                     "is linked to interview topics that exist in the question dataset; topics are ranked by how many of your skills "
                     "point to them. This is keyword matching, so it can miss skills that are not in the dictionary and cannot judge "
                     "how well you know a skill."),
                warning=None if skills else "No known skills were detected. The resume may use uncommon wording or be mostly images. "
                                            "You can still choose a topic manually in Interview Questions.")


def skill_catalog():
    """All skills the dictionary understands, grouped (used by the 'add a skill' control)."""
    return {g: sorted(k for k, v in SKILLS.items() if v["group"] == g) for g in GROUP_ORDER}


def rebuild(parsed, skills, pair_counts):
    """Apply the student's corrections: keep only dictionary skills, regroup and recompute recommended topics."""
    skills = sorted({k for k in skills if k in SKILLS})
    by_group = {g: [k for k in skills if SKILLS[k]["group"] == g] for g in GROUP_ORDER}
    parsed.update(skills=skills, skill_count=len(skills), skills_by_group={g: v for g, v in by_group.items() if v},
                  recommended_topics=recommend(skills, pair_counts), edited=True,
                  warning=None if skills else "No skills selected. Add skills above or choose a topic manually in Interview Questions.")
    return parsed
