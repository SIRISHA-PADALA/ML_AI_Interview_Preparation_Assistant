import io, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import db
db.DB_PATH = __import__("pathlib").Path(tempfile.mkdtemp()) / "t.db"
import app as A
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

c = A.app.test_client()
S = "test-session"


def make_pdf():
    b = io.BytesIO(); p = canvas.Canvas(b, pagesize=A4); y = 800
    for l in ["Asha Rao", "Skills", "Python, Docker, Kubernetes, React, SQL, Git, Machine Learning", "Education",
              "B.Tech Computer Science, JNTU University, CGPA 8.4", "Projects", "Interview Twin - Flask and React app",
              "Experience", "Software intern at Acme, 1 year of experience with CI/CD pipelines"]:
        p.drawString(60, y, l); y -= 24
    p.save(); return b.getvalue()


def test_filters_and_dataset():
    j = c.get("/api/filters").get_json(); assert len(j["topics"]) > 20
    d = c.get("/api/datasets").get_json(); assert d["total_usable"] > 3000 and len(d["sources"]) == 2
    assert c.get("/api/datasets/rows?status=needs_review").get_json()["total"] > 1000


def test_question_flow_and_adaptive():
    assert c.get("/api/questions/next").status_code == 400
    q = c.get(f"/api/questions/next?session_id={S}&topic=Docker&difficulty=Easy").get_json()["question"]
    assert q["topic"] == "Docker" and "?" in q["question"] or q["question"]
    assert c.post("/api/evaluate", json={"session_id": S, "uid": q["uid"], "answer": "x"}).status_code == 400
    ref = A.POOL[A.POOL.uid == q["uid"]].iloc[0].answer
    good = c.post("/api/evaluate", json={"session_id": S, "uid": q["uid"], "answer": ref}).get_json()
    bad = c.post("/api/evaluate", json={"session_id": S, "uid": "Q99999", "answer": "something here"})
    assert bad.status_code == 404 and good["result"]["score"] >= 75, good
    q2 = c.get(f"/api/questions/next?session_id={S}&topic=Docker").get_json()["question"]
    assert q2["uid"] != q["uid"]
    off = c.post("/api/evaluate", json={"session_id": S, "uid": q2["uid"], "answer": "I like pizza and football a lot"}).get_json()
    assert off["result"]["score"] < 30, off
    a = c.get(f"/api/questions/next?session_id={S}&adaptive=1").get_json()
    assert a["difficulty_reason"]


def test_history_dashboard():
    h = c.get(f"/api/history?session_id={S}").get_json()["attempts"]; assert len(h) == 2 and h[0]["created_at"]
    d = c.get(f"/api/dashboard?session_id={S}").get_json(); assert d["total"] == 2 and d["recommendations"]
    assert c.get("/api/dashboard?session_id=nobody").get_json()["total"] == 0


def upload(data, name="cv.pdf", sid=S):
    return c.post("/api/resume", data={"session_id": sid, "resume": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_resume_validation():
    assert c.post("/api/resume", data={"session_id": S}).status_code == 400            # no file
    assert "Only PDF" in upload(b"x", "cv.docx").get_json()["error"]                    # wrong type
    assert "empty" in upload(b"", "cv.pdf").get_json()["error"]                         # empty file
    assert "not a valid PDF" in upload(b"hello world", "cv.pdf").get_json()["error"]    # fake pdf
    big = upload(b"%PDF-" + b"0" * (6 * 1024 * 1024))                                   # > 5 MB
    assert big.status_code == 413 and "too large" in big.get_json()["error"]
    blank = io.BytesIO(); pc = canvas.Canvas(blank, pagesize=A4); pc.showPage(); pc.save()   # PDF with no text
    r = upload(blank.getvalue()); assert r.status_code == 400 and "No readable text" in r.get_json()["error"]
    assert "Could not read" in upload(b"%PDF-1.4 garbage").get_json()["error"]          # corrupted pdf


def test_resume_extraction_and_skills():
    r = upload(make_pdf()); j = r.get_json(); assert r.status_code == 200, j
    assert j["filename"] == "cv.pdf" and "Asha Rao" in j["text"] and "cleaned_text" not in j
    assert {"python", "docker", "kubernetes", "react", "sql", "git", "machine learning", "ci/cd"} <= set(j["skills"])
    assert len(j["skills"]) == len(set(j["skills"])) == j["skill_count"]                # no duplicates
    assert "Cloud & DevOps" in j["skills_by_group"] and j["how"] and j["education"] and j["projects"]


def test_cleaning_and_matching():
    import resume as R
    t = R.clean_text("See https://github.com/x and a@b.com. Built APIs in C++, Node.js, C#, CI/CD & MySQL (not JavaScript)!")
    assert "http" not in t and "a@b.com" not in t and all(k in t for k in ["c++", "node.js", "c#", "ci/cd", "mysql"])
    sk = R.find_skills(t)
    assert {"c++", "node.js", "c#", "ci/cd", "mysql"} <= set(sk) and "java" not in sk and "sql" not in sk and "javascript" in sk   # no substring false hits
    assert R.find_skills(R.clean_text("I enjoy cooking and football")) == []


def test_every_skill_target_exists_in_dataset():
    import resume as R
    bad = [(k, p) for k, v in R.SKILLS.items() for p in v["targets"] if p not in A.PAIR_COUNTS]
    assert not bad, bad


def test_recommendations_use_real_topics():
    j = upload(make_pdf()).get_json(); recs = j["recommended_topics"]; assert recs
    for r in recs:
        assert (r["category"], r["topic"]) in A.PAIR_COUNTS and r["because"] and r["questions_available"] > 0
    pairs = {(r["category"], r["topic"]) for r in recs}
    assert ("DevOps Engineer", "Docker") in pairs and ("React Engineer", "Hooks") in pairs and ("AI & ML", "ML") in pairs
    assert ("Software Interview Bank", "Database and SQL") in pairs and ("AI & ML", "Python") in pairs


def test_no_skills_gives_helpful_message():
    b = io.BytesIO(); p = canvas.Canvas(b, pagesize=A4)
    p.drawString(60, 800, "I enjoy cooking, football and travelling with friends every single weekend.")
    p.save(); j = upload(b.getvalue()).get_json()
    assert j["skills"] == [] and j["recommended_topics"] == [] and "manually" in j["warning"]


def test_start_interview_from_recommendation():
    j = upload(make_pdf()).get_json()
    rec = next(r for r in j["recommended_topics"] if r["topic"] == "Docker")
    for diff in ("Easy", "Medium", "Hard"):    # exactly what the UI sends when a topic + difficulty are picked
        q = c.get("/api/questions/next", query_string=dict(session_id="fresh-" + diff, category=rec["category"],
                  topic=rec["topic"], difficulty=diff, use_resume=1)).get_json()["question"]
        assert q["topic"] == "Docker" and q["category"] == "DevOps Engineer" and q["difficulty"] == diff
    m = c.get("/api/questions/next", query_string=dict(session_id=S, use_resume=1)).get_json()   # mixed resume interview
    assert (m["question"]["category"], m["question"]["topic"]) in {(r["category"], r["topic"]) for r in j["recommended_topics"]}
    assert "resume-recommended" in m["difficulty_reason"]


def test_resume_persists_and_delete_and_no_file_stored():
    upload(make_pdf(), sid="keep")
    got = c.get("/api/resume?session_id=keep").get_json()["resume"]; assert got["filename"] == "cv.pdf" and "text" not in got
    assert c.delete("/api/resume?session_id=keep").get_json()["deleted"]
    assert c.get("/api/resume?session_id=keep").get_json()["resume"] is None
    import pathlib; assert not list(pathlib.Path(A.ROOT).rglob("*.pdf"))                 # nothing written to disk


def test_page():
    assert b"Interview" in c.get("/").data


def test_every_detected_skill_has_a_topic():
    j = upload(make_pdf()).get_json()
    covered = {s for r in j["recommended_topics"] for s in r["because"]}
    assert set(j["skills"]) <= covered
