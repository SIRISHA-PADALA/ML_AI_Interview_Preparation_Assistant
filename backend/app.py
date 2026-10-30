import io, json, random, re, secrets
from pathlib import Path
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory, session
from werkzeug.security import check_password_hash, generate_password_hash
import db, report, resume as rs
import joblib
from nlp import Evaluator
from train_model import MODEL_DIR, preprocess, train as train_difficulty_model

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
_key = ROOT / "data" / ".secret_key"          # persisted so logins survive restarts; never committed/zipped
if not _key.exists():
    _key.write_text(secrets.token_hex(32))
app.secret_key = _key.read_text()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
LEVELS = ["Easy", "Medium", "Hard"]


def load_data():
    p = ROOT / "data" / "processed" / "unified_questions.csv"
    if not p.exists():
        from data_prep import build
        build()
    df = pd.read_csv(p).fillna("")
    return df


DF = load_data()
POOL = DF[DF.status == "converted"].reset_index(drop=True)
EVAL = Evaluator((POOL.answer).tolist())
REPORT = json.loads((ROOT / "data" / "processed" / "build_report.json").read_text())
PAIR_COUNTS = {(c, t): int(n) for (c, t), n in POOL.groupby(["category", "topic"]).size().items()}


def load_ml():
    """Load the saved difficulty model; train it first if the files are missing."""
    files = [MODEL_DIR / n for n in ("tfidf_vectorizer.joblib", "difficulty_model.joblib", "metrics.json")]
    if not all(f.exists() for f in files):
        train_difficulty_model(MODEL_DIR)
    return joblib.load(files[0]), joblib.load(files[1]), json.loads(files[2].read_text())


ML_VEC, ML_MODEL, ML_METRICS = load_ml()
_CLUSTER_CACHE = {}


def err(msg, code=400):
    return jsonify(error=msg), code


def resolve(raw):
    """Logged-in users always use their account id; guests use the random id their browser generated."""
    if session.get("uid"):
        return f"user{session['uid']}"
    raw = (raw or "").strip()
    return raw[:64] if raw else None


def sid():
    return resolve(request.values.get("session_id"))


def public(q):
    return dict(uid=q.uid, topic=q.topic, category=q.category, difficulty=q.difficulty, question=q.question,
                cluster=q.cluster_label)


def adaptive(c, s):
    last = [a["score"] for a in db.attempts(c, s, 3)]
    prev = db.attempts(c, s, 1)
    if not last:
        return "Medium", "No history yet, so starting at Medium."
    cur = prev[0]["difficulty"]
    avg = sum(last) / len(last)
    i = LEVELS.index(cur) if cur in LEVELS else 1
    if avg >= 70 and i < 2:
        return LEVELS[i + 1], f"Your last {len(last)} answer(s) averaged {avg:.0f}/100, so difficulty goes up from {cur}."
    if avg <= 40 and i > 0:
        return LEVELS[i - 1], f"Your last {len(last)} answer(s) averaged {avg:.0f}/100, so difficulty goes down from {cur}."
    return cur, f"Your recent average is {avg:.0f}/100, so staying at {cur}."


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/filters")
def filters():
    return jsonify(categories=sorted(POOL.category.unique()), difficulties=LEVELS,
                   topics=[dict(topic=t, category=c, count=int(n)) for (t, c), n in POOL.groupby(["topic", "category"]).size().items()])


@app.get("/api/questions/next")
def next_question():
    s = sid()
    if not s:
        return err("session_id is required")
    d = POOL
    topic, cat, diff = request.args.get("topic"), request.args.get("category"), request.args.get("difficulty")
    why = None
    with db.conn() as c:
        if request.args.get("adaptive") == "1":
            diff, why = adaptive(c, s)
        elif diff and diff not in LEVELS:
            return err("difficulty must be Easy, Medium or Hard")
        seen = {a["uid"] for a in db.attempts(c, s)}
        resume = db.get_resume(c, s) if request.args.get("use_resume") == "1" else None
    if topic:
        d = d[d.topic == topic]
    if cat:
        d = d[d.category == cat]
    d = d[~d.uid.isin(seen)]
    note = why
    if diff:
        sub = d[d.difficulty == diff]
        if sub.empty and not d.empty:
            near = sorted(LEVELS, key=lambda x: abs(LEVELS.index(x) - LEVELS.index(diff)))[1:]
            for n in near:
                sub = d[d.difficulty == n]
                if not sub.empty:
                    note = (why + " " if why else "") + f"No unseen {diff} questions match, so showing {n}."
                    break
        d = sub
    if request.args.get("use_resume") == "1" and not resume:
        note = (note + " " if note else "") + "No resume uploaded, so showing general questions."
    if resume and not topic:
        rec = {(r.get("category"), r["topic"]) for r in resume["recommended_topics"]}
        pref = d[[(c, t) in rec for c, t in zip(d.category, d.topic)]]
        if not pref.empty:
            d = pref
            note = (note + " " if note else "") + "Question chosen from your resume-recommended topics."
    if d.empty:
        return jsonify(question=None, message="No unseen questions match these filters. Try other filters or review your history."), 200
    q = d.iloc[random.randrange(len(d))]
    return jsonify(question=public(q), difficulty_reason=note, remaining=int(len(d) - 1))


@app.post("/api/evaluate")
def evaluate():
    b = request.get_json(silent=True) or {}
    s, uid, ans = resolve(b.get("session_id")), b.get("uid"), (b.get("answer") or "").strip()
    if not s or not uid:
        return err("session_id and uid are required")
    if len(ans) < 3:
        return err("Please type an answer before submitting.")
    if len(ans) > 4000:
        return err("Answer is too long (max 4000 characters).")
    row = POOL[POOL.uid == uid]
    if row.empty:
        return err("Unknown question.", 404)
    q = row.iloc[0]
    r = EVAL.evaluate(q.question, q.answer, ans)
    tips = list(r["notes"])
    if r["missing_concepts"]:
        tips.append("Try to cover: " + ", ".join(r["missing_concepts"][:5]) + ".")
    if r["label"] == "Strong":
        tips.append("Good coverage of the reference answer. Add a real-world example to stand out.")
    tips.append("Remember: this score compares wording with the reference; it cannot confirm your answer is technically correct.")
    mock_id = b.get("mock_id")
    with db.conn() as c:
        if mock_id is not None:
            m = c.execute("SELECT * FROM mock_sessions WHERE id=? AND session_id=?", (mock_id, s)).fetchone()
            if not m or uid not in json.loads(m["uids"]):
                return err("This question is not part of that mock interview.")
            if m["ended_at"]:
                return err("This mock interview has already ended.")
            if c.execute("SELECT 1 FROM attempts WHERE mock_id=? AND uid=?", (mock_id, uid)).fetchone():
                return err("You already answered this question in this mock interview.")
        db.save_attempt(c, session_id=s, uid=uid, topic=q.topic, category=q.category, difficulty=q.difficulty,
                        question=q.question, user_answer=ans, reference=q.answer, score=r["score"], label=r["label"],
                        feedback=tips, missing=r["missing_concepts"], covered=r["covered_concepts"], mock_id=mock_id)
    return jsonify(result=r, suggestions=tips, reference_answer=q.answer)


@app.get("/api/history")
def history():
    s = sid()
    if not s:
        return err("session_id is required")
    with db.conn() as c:
        rows = db.attempts(c, s)
    for r in rows:
        r["feedback"], r["missing"], r["covered"] = json.loads(r["feedback"]), json.loads(r["missing"]), json.loads(r.get("covered") or "[]")
    return jsonify(attempts=rows)


@app.get("/api/dashboard")
def dashboard():
    s = sid()
    if not s:
        return err("session_id is required")
    with db.conn() as c:
        rows = db.attempts(c, s)
    if not rows:
        return jsonify(total=0, topics=[], weak=[], strong=[], by_difficulty=[], recent=[], trend=[], completed_mocks=0, frequent_topics=[], frequent_missing=[], heatmap=dict(topics=[], difficulties=LEVELS, cells=[]), progress=dict(attempted=0, total=len(POOL)), suggested_next=["Start a short practice session or a mock interview."], recommendations=["Answer a few questions to see your analysis."])
    df = pd.DataFrame(rows)
    g = df.groupby("topic").agg(avg=("score", "mean"), attempts=("score", "size")).reset_index()
    topics = [dict(topic=r.topic, avg=round(r.avg), attempts=int(r.attempts)) for r in g.sort_values("avg").itertuples()]
    weak = [t for t in topics if t["avg"] < 60]
    miss = {}
    for r in rows:
        for m in json.loads(r["missing"]):
            miss.setdefault(r["topic"], {}).setdefault(m, 0)
            miss[r["topic"]][m] += 1
    recs = []
    for t in weak[:5]:
        top = sorted(miss.get(t["topic"], {}).items(), key=lambda x: -x[1])[:4]
        recs.append(f"Revise {t['topic']} (avg {t['avg']}/100 over {t['attempts']} answer(s))" +
                    (f"; often-missed concepts: {', '.join(k for k, _ in top)}." if top else "."))
    if not weak:
        recs.append("No weak topics yet. Try Hard questions or new topics.")
    if len(df) < 5:
        recs.append("Fewer than 5 answers so far: treat these results as early indicators.")
    trend = [dict(n=i + 1, score=int(x)) for i, x in enumerate(df.score[::-1])]
    strong = [t for t in topics if t["avg"] >= 75][::-1]
    gd = df.groupby("difficulty").agg(avg=("score", "mean"), attempts=("score", "size"))
    by_diff = [dict(difficulty=d, avg=round(gd.loc[d, "avg"]), attempts=int(gd.loc[d, "attempts"])) for d in LEVELS if d in gd.index]
    recent = [dict(question=r["question"], topic=r["topic"], difficulty=r["difficulty"], score=r["score"], label=r["label"], created_at=r["created_at"]) for r in rows[:6]]
    mm = {}
    for r in rows:
        for k in json.loads(r["missing"]):
            mm[k] = mm.get(k, 0) + 1
    with db.conn() as c:
        done = c.execute("SELECT COUNT(DISTINCT m.id) FROM mock_sessions m JOIN attempts a ON a.mock_id=m.id WHERE m.session_id=? AND m.ended_at IS NOT NULL", (s,)).fetchone()[0]
    pv = df.pivot_table(index="topic", columns="difficulty", values="score", aggfunc="mean")
    heat = dict(topics=list(pv.index), difficulties=LEVELS, cells=[[None if (d not in pv.columns or pd.isna(pv.loc[t, d])) else round(float(pv.loc[t, d])) for d in LEVELS] for t in pv.index])
    nxt = ["Re-attempt " + w["topic"] + " questions at an easier level, then work back up." for w in weak[:2]]
    nxt += ["Try a Hard question in " + t["topic"] + " (you average " + str(t["avg"]) + ")." for t in strong[:1]]
    nxt.append("Take a mock interview to practise answering under realistic conditions." if done == 0 else "Review your last mock interview report and retry its weakest topics.")
    freq = sorted(topics, key=lambda t: -t["attempts"])[:5]
    return jsonify(total=len(df), average=round(df.score.mean()), topics=topics, weak=weak, strong=strong, by_difficulty=by_diff,
                   recent=recent, trend=trend, recommendations=recs, completed_mocks=int(done), frequent_topics=freq,
                   frequent_missing=[dict(concept=k, count=v) for k, v in sorted(mm.items(), key=lambda x: -x[1])[:10]],
                   heatmap=heat, progress=dict(attempted=int(df.uid.nunique()), total=len(POOL)), suggested_next=nxt)


@app.errorhandler(Exception)
def unhandled(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    app.logger.exception("Unhandled error")          # details go to the server log only, never to the browser
    return err("Something went wrong on the server. Please try again.", 500)


@app.errorhandler(413)
def too_big(_):
    return err("File is too large. The maximum size is 5 MB.", 413)


@app.post("/api/resume")
def resume_upload():
    s, f = sid(), request.files.get("resume")
    if not s:
        return err("session_id is required")
    if not f or not f.filename:
        return err("Choose a PDF file first.")
    if not f.filename.lower().endswith(".pdf"):
        return err("Only PDF files are accepted. Please upload your resume as a .pdf file.")
    data = f.read()  # kept in memory only; the uploaded file is never written to disk
    if not data:
        return err("The file is empty.")
    if not data.startswith(b"%PDF"):
        return err("This file is not a valid PDF (renaming another file to .pdf does not work).")
    try:
        text = rs.extract_text(io.BytesIO(data))
    except Exception:
        return err("Could not read this PDF. It may be corrupted or password-protected.")
    if len(text) < 40:
        return err("No readable text found. Scanned or image-only resumes are not supported (no OCR).")
    name = Path(f.filename).name[:120]
    parsed = rs.analyze(text, PAIR_COUNTS, name)
    with db.conn() as c:
        db.save_resume(c, s, name, parsed)  # only the analysis result is stored, not the file or its text
    out = dict(parsed, text=text[:20000], text_truncated=len(text) > 20000)
    if request.args.get("debug") == "1":
        out["cleaned_text"] = rs.clean_text(text)[:20000]
    return jsonify(out)


@app.get("/api/resume")
def resume_get():
    s = sid()
    with db.conn() as c:
        r = db.get_resume(c, s) if s else None
    return jsonify(resume=r, skill_options=rs.skill_catalog(), message=None if r else "Upload your resume to get personalized recommendations.")


@app.get("/api/resume/questions")
def resume_questions():
    s = sid()
    if not s:
        return err("session_id is required")
    with db.conn() as c:
        resume = db.get_resume(c, s)
        seen = {a["uid"] for a in db.attempts(c, s)}
    if not resume:
        return jsonify(has_resume=False, message="Upload your resume to get personalized recommendations.", questions=[])
    out, d = [], POOL[~POOL.uid.isin(seen)]
    for r in resume["recommended_topics"]:
        sub = d[(d.category == r.get("category")) & (d.topic == r["topic"])]
        if len(sub):
            q = sub.iloc[random.randrange(len(sub))]
            out.append(dict(public(q), because=r["because"]))
        if len(out) >= 8:
            break
    return jsonify(has_resume=True, questions=out)


@app.delete("/api/resume")
def resume_delete():
    s = sid()
    if not s:
        return err("session_id is required")
    with db.conn() as c:
        c.execute("DELETE FROM resumes WHERE session_id=?", (s,))
        c.commit()
    return jsonify(deleted=True)


@app.get("/api/datasets")
def datasets():
    out = []
    for src, g in DF.groupby("source"):
        ok = g[g.status == "converted"]
        out.append(dict(source=src, raw_rows=REPORT["files"][src]["rows"], columns=REPORT["files"][src]["columns"],
                        encoding=REPORT["files"][src]["encoding"], status_counts=g.status.value_counts().to_dict(),
                        categories=ok.category.value_counts().to_dict(), difficulty=ok.difficulty.value_counts().to_dict(),
                        topics=int(ok.topic.nunique()), top_topics=ok.topic.value_counts().head(12).to_dict()))
    cl = POOL.groupby(["cluster_id", "cluster_label"]).size().reset_index(name="n")
    pre = ["Read both CSVs (encoding auto-detected), normalised Unicode and whitespace",
           "Converted MCQs to open-ended Q&A using the dataset's marked correct option (nothing invented)",
           f"Flagged {int((DF.status == 'duplicate').sum())} duplicate questions and {int((DF.status == 'needs_review').sum())} rows needing review (aptitude, option-dependent, or suspicious answer key)",
           f"Kept {len(POOL)} usable questions; grouped them into 14 K-Means topic clusters"]
    return jsonify(sources=out, total_usable=int(len(POOL)), preprocessing=pre, quality=REPORT["quality"],
                   clusters=[dict(id=int(r.cluster_id), label=r.cluster_label, count=int(r.n)) for r in cl.itertuples()],
                   method_notes=["Scores use TF-IDF + cosine similarity and key-concept coverage: lexical matching, not semantic understanding.",
                                 "MCQ conversion uses the dataset's marked correct option as the reference answer; no explanations were invented.",
                                 "K-Means clusters group questions by shared vocabulary and are only a rough guide."])


@app.get("/api/ml/metrics")
def ml_metrics():
    return jsonify(ML_METRICS)


@app.post("/api/ml/predict")
def ml_predict():
    q = ((request.get_json(silent=True) or {}).get("question") or "").strip()
    if len(q) < 8:
        return err("Enter a question of at least 8 characters.")
    if len(q) > 1000:
        return err("Question is too long (max 1000 characters).")
    X = ML_VEC.transform([preprocess(q)])
    p = ML_MODEL.predict_proba(X)[0]
    probs = {c: round(float(p[list(ML_MODEL.classes_).index(c)]), 3) for c in ["Easy", "Medium", "Hard"]}
    label = max(probs, key=probs.get)
    known = int(X.nnz)
    return jsonify(predicted=label, confidence=probs[label], probabilities=probs, known_terms=known,
                   warning=None if known >= 2 else "Few or no words in this question were seen in training, so this prediction is unreliable.",
                   note="Trained on this project's labelled questions; a rough hint, not a ground-truth difficulty.")


@app.get("/api/datasets/clusters")
def cluster_plot():
    if not _CLUSTER_CACHE:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from nlp import tokens
        smp = POOL.sample(min(700, len(POOL)), random_state=42)
        X = TfidfVectorizer(analyzer=tokens, sublinear_tf=True, min_df=2).fit_transform(smp.question + " " + smp.answer)
        xy = TruncatedSVD(2, random_state=42).fit_transform(X)
        labels = POOL.drop_duplicates("cluster_id").set_index("cluster_id").cluster_label.to_dict()
        _CLUSTER_CACHE.update(points=[dict(x=round(float(a), 4), y=round(float(b), 4), c=int(c)) for (a, b), c in zip(xy, smp.cluster_id)],
                              clusters=[dict(id=int(k), label=v) for k, v in sorted(labels.items())],
                              note="K-Means (k=14) was fitted on all usable questions. This map is a 2-D SVD projection of a 700-question sample, so clusters overlap visually.")
    return jsonify(_CLUSTER_CACHE)


@app.get("/api/datasets/rows")
def dataset_rows():
    st, src = request.args.get("status", "converted"), request.args.get("source")
    d = DF[DF.status == st]
    if src:
        d = d[d.source == src]
    n = min(int(request.args.get("limit", 15)), 100)
    d = d.sample(min(n, len(d)), random_state=random.randrange(10**6)) if len(d) else d
    return jsonify(total=int(len(DF[DF.status == st])), rows=d[["uid", "source", "src_id", "category", "topic", "difficulty", "question", "answer", "note"]].to_dict("records"))


# ----------------------------- accounts -----------------------------
USER_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")


def _claim(c, guest, user_key):
    """Move work done as a guest into the account the student just created / signed in to."""
    guest = (guest or "").strip()[:64]
    if guest and guest != user_key and not guest.startswith("user"):
        c.execute("UPDATE attempts SET session_id=? WHERE session_id=?", (user_key, guest))
        c.execute("UPDATE mock_sessions SET session_id=? WHERE session_id=?", (user_key, guest))
        if not c.execute("SELECT 1 FROM resumes WHERE session_id=?", (user_key,)).fetchone():
            c.execute("UPDATE resumes SET session_id=? WHERE session_id=?", (user_key, guest))
        c.commit()


@app.post("/api/auth/register")
def register():
    b = request.get_json(silent=True) or {}
    u, pw = (b.get("username") or "").strip(), b.get("password") or ""
    if not USER_RE.match(u):
        return err("Username must be 3-30 characters: letters, numbers, dot, dash or underscore.")
    if len(pw) < 8 or len(pw) > 128:
        return err("Password must be 8-128 characters.")
    with db.conn() as c:
        try:
            cur = c.execute("INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)", (u, generate_password_hash(pw), db.now()))
            c.commit()
        except Exception:
            return err("That username is already taken.", 409)
        session["uid"], session["username"] = cur.lastrowid, u
        _claim(c, b.get("session_id"), f"user{cur.lastrowid}")
    return jsonify(user=dict(username=u))


@app.post("/api/auth/login")
def login():
    b = request.get_json(silent=True) or {}
    u, pw = (b.get("username") or "").strip(), b.get("password") or ""
    with db.conn() as c:
        r = c.execute("SELECT * FROM users WHERE username=?", (u,)).fetchone()
        if not r or not check_password_hash(r["password_hash"], pw):
            return err("Invalid username or password.", 401)
        session["uid"], session["username"] = r["id"], r["username"]
        _claim(c, b.get("session_id"), f"user{r['id']}")
    return jsonify(user=dict(username=r["username"]))


@app.post("/api/auth/logout")
def logout():
    session.clear()
    return jsonify(user=None)


@app.get("/api/auth/me")
def me():
    return jsonify(user=dict(username=session["username"]) if session.get("uid") else None)


# --------------------------- mock interviews ---------------------------
def mock_summary(c, mid, s):
    m = c.execute("SELECT * FROM mock_sessions WHERE id=? AND session_id=?", (mid, s)).fetchone()
    if not m:
        return None
    rows = [dict(r) for r in c.execute("SELECT * FROM attempts WHERE mock_id=? ORDER BY id", (mid,))]
    for r in rows:
        r["missing"], r["covered"], r["feedback"] = json.loads(r["missing"]), json.loads(r["covered"] or "[]"), json.loads(r["feedback"])
    sc = [r["score"] for r in rows]
    miss = {}
    for r in rows:
        for k in r["missing"]:
            miss[k] = miss.get(k, 0) + 1
    by = {}
    for r in rows:
        by.setdefault(r["topic"], []).append(r["score"])
    return dict(id=m["id"], total_questions=m["count"], answered=len(rows), average=round(sum(sc) / len(sc)) if sc else None,
                ended=bool(m["ended_at"]), started_at=m["started_at"], ended_at=m["ended_at"], timer_seconds=m["timer_seconds"],
                topics=[dict(topic=t, avg=round(sum(v) / len(v)), attempts=len(v)) for t, v in by.items()],
                frequent_missing=[dict(concept=k, count=v) for k, v in sorted(miss.items(), key=lambda x: -x[1])[:8]],
                answers=[dict(uid=r["uid"], question=r["question"], topic=r["topic"], difficulty=r["difficulty"], score=r["score"], label=r["label"],
                              missing=r["missing"], covered=r["covered"]) for r in rows],
                unanswered=max(0, m["count"] - len(rows)))


@app.post("/api/mock/start")
def mock_start():
    b = request.get_json(silent=True) or {}
    s = resolve(b.get("session_id"))
    if not s:
        return err("session_id is required")
    try:
        n = int(b.get("count", 5)); timer = int(b.get("timer_seconds") or 0)
    except (TypeError, ValueError):
        return err("count and timer_seconds must be numbers.")
    if not 1 <= n <= 20:
        return err("Choose between 1 and 20 questions.")
    if timer and not 30 <= timer <= 3600:
        return err("Timer must be off (0) or between 30 and 3600 seconds per question.")
    diff, topic, cat = b.get("difficulty") or "", b.get("topic") or "", b.get("category") or ""
    if diff and diff not in LEVELS:
        return err("difficulty must be Easy, Medium or Hard (or empty for mixed).")
    with db.conn() as c:
        seen = {a["uid"] for a in db.attempts(c, s)}
        resume = db.get_resume(c, s) if b.get("use_resume") else None
    d = POOL[~POOL.uid.isin(seen)]
    if cat: d = d[d.category == cat]
    if topic: d = d[d.topic == topic]
    if diff: d = d[d.difficulty == diff]
    if resume and not topic:
        rec = {(r.get("category"), r["topic"]) for r in resume["recommended_topics"]}
        pref = d[[(x, y) in rec for x, y in zip(d.category, d.topic)]]
        d = pref if len(pref) >= n else d
    if d.empty:
        return err("No unseen questions match these choices. Try other topics or a different difficulty.")
    pick = d.sample(min(n, len(d)), random_state=random.randrange(10**6))
    with db.conn() as c:
        cur = c.execute("INSERT INTO mock_sessions(session_id,category,topic,difficulty,count,timer_seconds,uids,started_at) VALUES(?,?,?,?,?,?,?,?)",
                        (s, cat, topic, diff, len(pick), timer, json.dumps(list(pick.uid)), db.now()))
        c.commit()
    return jsonify(mock_id=cur.lastrowid, timer_seconds=timer, questions=[public(q) for q in pick.itertuples()],
                   note=None if len(pick) == n else f"Only {len(pick)} unseen question(s) matched, so the interview has {len(pick)}.")


@app.get("/api/mock/<int:mid>")
def mock_get(mid):
    s = sid()
    with db.conn() as c:
        out = mock_summary(c, mid, s) if s else None
    return jsonify(out) if out else err("Mock interview not found.", 404)


@app.post("/api/mock/<int:mid>/end")
def mock_end(mid):
    s = resolve((request.get_json(silent=True) or {}).get("session_id"))
    with db.conn() as c:
        if not s or not c.execute("SELECT 1 FROM mock_sessions WHERE id=? AND session_id=?", (mid, s)).fetchone():
            return err("Mock interview not found.", 404)
        c.execute("UPDATE mock_sessions SET ended_at=COALESCE(ended_at,?) WHERE id=?", (db.now(), mid))
        c.commit()
        return jsonify(mock_summary(c, mid, s))


@app.get("/api/mock")
def mock_list():
    s = sid()
    if not s:
        return err("session_id is required")
    with db.conn() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM mock_sessions WHERE session_id=? ORDER BY id DESC LIMIT 20", (s,))]
        sums = [mock_summary(c, i, s) for i in ids]
    return jsonify(mocks=[dict((k, m[k]) for k in ("id", "total_questions", "answered", "average", "ended", "started_at")) for m in sums])


# ---------------------------- recommendations ----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlp import tokens
QVEC = TfidfVectorizer(analyzer=tokens, sublinear_tf=True, min_df=2).fit(POOL.question)
QX = QVEC.transform(POOL.question)


@app.get("/api/recommend")
def recommend_questions():
    s = sid()
    if not s:
        return err("session_id is required")
    limit = max(1, min(int(request.args.get("limit", 6) or 6), 15))
    with db.conn() as c:
        rows = db.attempts(c, s, 500)
        resume = db.get_resume(c, s)
        diff, why = adaptive(c, s)
    if request.args.get("difficulty") in LEVELS:
        diff, why = request.args["difficulty"], "You chose this difficulty."
    seen = {r["uid"] for r in rows}
    d = POOL[~POOL.uid.isin(seen)]
    sub = d[d.difficulty == diff]
    d = sub if len(sub) >= limit else d
    weak = {}
    if rows:
        g = pd.DataFrame(rows).groupby("topic").score.mean()
        weak = {t: round(v) for t, v in g.items() if v < 60}
    rec = {(r.get("category"), r["topic"]): r["because"] for r in (resume or {}).get("recommended_topics", [])}
    low = [r["question"] for r in rows if r["score"] < 60][:30]
    sim = cosine_similarity(QX[d.index], QVEC.transform(low)).max(axis=1) if low and len(d) else [0.0] * len(d)
    score, tags = [], []
    for (i, q), sm in zip(d.iterrows(), sim):
        t, why_q = 0.0, []
        if q.topic in weak:
            t += 1.0; why_q.append(f"weak topic ({q.topic}, your average is {weak[q.topic]})")
        if (q.category, q.topic) in rec:
            t += 0.5; why_q.append("matches your resume skills (" + ", ".join(rec[(q.category, q.topic)]) + ")")
        if sm >= 0.2:
            t += 1.5 * float(sm); why_q.append(f"wording similar to a question you scored low on (cosine {sm:.2f})")
        score.append(t + random.random() * 0.05); tags.append(why_q)
    order = sorted(range(len(d)), key=lambda k: -score[k])
    chosen, out = [], []
    for k in order:
        i = d.index[k]
        if chosen and cosine_similarity(QX[i], QX[chosen]).max() > 0.8:      # duplicate / near-duplicate detection
            continue
        chosen.append(i)
        q = d.loc[i]
        out.append(dict(public(q), reason="; ".join(tags[k]) or "no history signal for this one, picked from unseen questions at the chosen level"))
        if len(out) >= limit:
            break
    personalized = bool(weak or rec or low)
    return jsonify(personalized=personalized, difficulty=diff, difficulty_reason=why, questions=out,
                   explanation=["Unseen only: questions you already answered are excluded.", "Topic matching: weak topics (+1.0) and resume-recommended topics (+0.5) rank higher.",
                                "TF-IDF + cosine similarity: questions worded like ones you scored low on rank higher (up to +1.5).",
                                "Difficulty: follows the adaptive rule (average of your last 3 scores), or the level you chose.",
                                "Duplicate detection: a question more than 0.8 cosine-similar to one already chosen is skipped."]
                   if personalized else ["No practice history or resume yet, so these are simply random unseen questions. Answer a few questions to get personalised ones."])


# ------------------------------ resume skills ------------------------------
@app.post("/api/resume/skills")
def resume_skills():
    b = request.get_json(silent=True) or {}
    s = resolve(b.get("session_id"))
    skills = b.get("skills")
    if not s or not isinstance(skills, list) or not all(isinstance(x, str) for x in skills):
        return err("session_id and a list of skills are required.")
    unknown = [x for x in skills if x not in rs.SKILLS]
    if unknown:
        return err("Not in the skill dictionary: " + ", ".join(unknown[:5]) + ". Pick skills from the list.")
    with db.conn() as c:
        cur = db.get_resume(c, s)
        if not cur:
            return err("Upload your resume first.", 404)
        parsed = rs.rebuild(cur, skills, PAIR_COUNTS)
        db.save_resume(c, s, parsed.get("filename", ""), parsed)
    return jsonify(parsed)


# --------------------------------- reports ---------------------------------
@app.get("/api/report.pdf")
def report_pdf():
    s = sid()
    if not s:
        return err("session_id is required")
    mid = request.args.get("mock_id", type=int)
    with db.conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM attempts WHERE session_id=?" + (" AND mock_id=?" if mid else "") + " ORDER BY id DESC", (s, mid) if mid else (s,))]
    if not rows:
        return err("No answers found yet, so there is nothing to put in a report.", 404)
    for r in rows:
        r["missing"], r["covered"] = json.loads(r["missing"]), json.loads(r["covered"] or "[]")
    pdf = report.build_pdf(rows, f"Interview Twin - {'Mock Interview #' + str(mid) if mid else 'Practice'} Report", session.get("username"))
    return Response(pdf, mimetype="application/pdf", headers={"Content-Disposition": "attachment; filename=interview-twin-report.pdf"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
