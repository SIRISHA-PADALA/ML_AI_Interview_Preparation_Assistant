import os, sys, sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import app as A, db, train_model as T
from test_api import make_pdf


def client():
    return A.app.test_client()          # each client has its own cookie jar


def ans(c, s, uid, text, **kw):
    return c.post("/api/evaluate", json=dict(session_id=s, uid=uid, answer=text, **kw))


def test_register_login_hashing_and_validation():
    c = client()
    assert c.post("/api/auth/register", json={"username": "ab", "password": "longenough1"}).status_code == 400
    assert c.post("/api/auth/register", json={"username": "asha_rao", "password": "short"}).status_code == 400
    assert c.post("/api/auth/register", json={"username": "bad name!", "password": "longenough1"}).status_code == 400
    r = c.post("/api/auth/register", json={"username": "asha_rao", "password": "S3cret-pass!"}); assert r.status_code == 200
    assert c.get("/api/auth/me").get_json()["user"]["username"] == "asha_rao"
    h = sqlite3.connect(db.DB_PATH).execute("SELECT password_hash FROM users WHERE username='asha_rao'").fetchone()[0]
    assert "S3cret-pass!" not in h and h.startswith(("scrypt:", "pbkdf2:"))                      # hashed, never plain text
    assert client().post("/api/auth/register", json={"username": "ASHA_RAO", "password": "S3cret-pass!"}).status_code == 409
    c.post("/api/auth/logout"); assert c.get("/api/auth/me").get_json()["user"] is None
    bad1 = client().post("/api/auth/login", json={"username": "asha_rao", "password": "wrong-pass"})
    bad2 = client().post("/api/auth/login", json={"username": "nobody", "password": "wrong-pass"})
    assert bad1.status_code == bad2.status_code == 401 and bad1.get_json() == bad2.get_json()    # no user enumeration
    assert client().post("/api/auth/login", json={"username": "asha_rao", "password": "S3cret-pass!"}).status_code == 200


def test_guest_work_is_claimed_and_account_isolated():
    g = client(); q = g.get("/api/questions/next?session_id=guestA&topic=Docker").get_json()["question"]
    assert ans(g, "guestA", q["uid"], "containers package applications").status_code == 200
    u = client(); u.post("/api/auth/register", json={"username": "ravi.k", "password": "S3cret-pass!", "session_id": "guestA"})
    assert len(u.get("/api/history?session_id=anything-else").get_json()["attempts"]) == 1      # account id wins over the client id
    assert client().get("/api/history?session_id=guestA").get_json()["attempts"] == []            # moved out of the guest session
    other = client(); other.post("/api/auth/register", json={"username": "other.u", "password": "S3cret-pass!"})
    assert other.get("/api/history?session_id=guestA").get_json()["attempts"] == []


def test_mock_interview_full_flow():
    c, s = client(), "mock1"
    for bad in (dict(count=0), dict(count=21), dict(count=5, timer_seconds=10), dict(count=5, difficulty="Nope")):
        assert c.post("/api/mock/start", json=dict(session_id=s, **bad)).status_code == 400
    j = c.post("/api/mock/start", json=dict(session_id=s, topic="Docker", count=4, timer_seconds=0)).get_json()
    qs, mid = j["questions"], j["mock_id"]; assert len(qs) == 4 and len({q["uid"] for q in qs}) == 4 and all(q["topic"] == "Docker" for q in qs)
    assert "answer" not in qs[0]                                                                  # reference answer is not leaked
    ref = A.POOL[A.POOL.uid == qs[0].get("uid")].iloc[0].answer
    assert ans(c, s, qs[0]["uid"], ref, mock_id=mid).status_code == 200
    assert ans(c, s, qs[0]["uid"], ref, mock_id=mid).status_code == 400                           # no double answer
    assert ans(c, s, "Q00001", "text here", mock_id=mid).status_code == 400                      # not part of this mock
    assert ans(c, "someone-else", qs[1]["uid"], "text here", mock_id=mid).status_code == 400      # not their mock
    assert c.get(f"/api/mock/{mid}?session_id=someone-else").status_code == 404
    summ = c.post(f"/api/mock/{mid}/end", json={"session_id": s}).get_json()
    assert summ["ended"] and summ["answered"] == 1 and summ["unanswered"] == 3 and summ["average"] >= 75
    assert ans(c, s, qs[1]["uid"], "late answer for ended mock", mock_id=mid).status_code == 400
    assert c.get(f"/api/dashboard?session_id={s}").get_json()["completed_mocks"] == 1
    assert c.get(f"/api/mock?session_id={s}").get_json()["mocks"][0]["id"] == mid
    m2 = c.post("/api/mock/start", json=dict(session_id=s, topic="Docker", count=4)).get_json()
    assert qs[0]["uid"] not in {q["uid"] for q in m2["questions"]}                                # answered questions are not repeated


def test_recommendations_explain_themselves():
    c, s = client(), "rec1"
    cold = c.get(f"/api/recommend?session_id={s}").get_json(); assert cold["personalized"] is False and "random" in cold["explanation"][0]
    for q in c.post("/api/mock/start", json=dict(session_id=s, topic="Kubernetes", count=3)).get_json()["questions"]:
        ans(c, s, q["uid"], "I like football")                                                    # deliberately poor answers
    r = c.get(f"/api/recommend?session_id={s}&limit=6").get_json(); assert r["personalized"] and len(r["questions"]) == 6
    done = {a["uid"] for a in c.get(f"/api/history?session_id={s}").get_json()["attempts"]}
    assert not done & {q["uid"] for q in r["questions"]}
    assert r["questions"][0]["topic"] == "Kubernetes" and "weak topic" in r["questions"][0]["reason"]


def test_resume_skill_correction():
    c, s = client(), "skills1"
    assert c.post("/api/resume/skills", json=dict(session_id=s, skills=["python"])).status_code == 404
    assert c.post("/api/resume", data={"session_id": s, "resume": (__import__("io").BytesIO(make_pdf()), "cv.pdf")}, content_type="multipart/form-data").status_code == 200
    assert c.post("/api/resume/skills", json=dict(session_id=s, skills=["python", "made-up-skill"])).status_code == 400
    j = c.post("/api/resume/skills", json=dict(session_id=s, skills=["docker"])).get_json()
    assert j["skills"] == ["docker"] and j["edited"] and {r["topic"] for r in j["recommended_topics"]} == {"Docker", "DevOps"}
    assert c.post("/api/resume/skills", json=dict(session_id=s, skills=[])).get_json()["warning"]
    assert "Cloud & DevOps" in c.get(f"/api/resume?session_id={s}").get_json()["skill_options"]


def test_pdf_report():
    c, s = client(), "pdf1"
    r = c.get(f"/api/report.pdf?session_id={s}"); assert r.status_code == 404 and "nothing to put" in r.get_json()["error"]
    q = c.get(f"/api/questions/next?session_id={s}&topic=Docker").get_json()["question"]; ans(c, s, q["uid"], "containers package apps with dependencies")
    r = c.get(f"/api/report.pdf?session_id={s}"); assert r.status_code == 200 and r.mimetype == "application/pdf" and r.data.startswith(b"%PDF") and len(r.data) > 1500
    m = c.post("/api/mock/start", json=dict(session_id=s, topic="Git", count=2)).get_json(); ans(c, s, m["questions"][0]["uid"], "version control tracks changes", mock_id=m["mock_id"])
    assert c.get(f"/api/report.pdf?session_id={s}&mock_id={m['mock_id']}").status_code == 200


def test_failure_handling(monkeypatch, tmp_path):
    c = client()
    def boom(*a, **k): raise sqlite3.OperationalError("database is locked: /secret/path.db")
    monkeypatch.setattr(db, "conn", boom)
    r = c.get("/api/history?session_id=x"); assert r.status_code == 500 and "sqlite" not in r.get_data(as_text=True).lower() and "secret" not in r.get_data(as_text=True)
    monkeypatch.undo()
    monkeypatch.setattr(A, "MODEL_DIR", tmp_path / "models")                                       # missing model -> trained automatically
    vec, model, metrics = A.load_ml(); assert (tmp_path / "models" / "difficulty_model.joblib").exists() and metrics["accuracy"] > 0
    try:
        T.load_data(tmp_path / "nope.csv"); assert False
    except FileNotFoundError:
        pass                                                                                       # missing dataset is a clear error
    bad = tmp_path / "bad.csv"; bad.write_text("uid,status,difficulty,question\nQ1,converted,Impossible,\nQ2,converted,Easy,   \n")
    assert len(T.load_data(bad)) == 0                                                              # invalid labels / empty text are dropped
