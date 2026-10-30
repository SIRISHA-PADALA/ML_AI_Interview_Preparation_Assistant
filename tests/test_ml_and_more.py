import os, sys, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import json, joblib
import app as A
import train_model as T
from test_api import make_pdf, upload

c = A.app.test_client()


def test_training_is_real_reproducible_and_saved(tmp_path):
    m1 = T.train(tmp_path); m2 = T.train(tmp_path)
    assert (tmp_path / "difficulty_model.joblib").exists() and (tmp_path / "tfidf_vectorizer.joblib").exists()
    assert m1["train_size"] + m1["test_size"] == m1["dataset_size"] and abs(m1["test_size"] / m1["dataset_size"] - 0.2) < 0.01
    assert m1["accuracy"] == m2["accuracy"] and m1["confusion_matrix"] == m2["confusion_matrix"]        # reproducible
    assert sum(map(sum, m1["confusion_matrix"])) == m1["test_size"]
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        assert 0 < m1[k] < 1
    assert m1["accuracy"] > m1["baseline_majority_accuracy"] and m1["f1_macro"] > m1["baseline_majority_f1_macro"]
    vec, model = joblib.load(tmp_path / "tfidf_vectorizer.joblib"), joblib.load(tmp_path / "difficulty_model.joblib")
    assert model.predict(vec.transform(["what is docker"]))[0] in T.LABELS


def test_training_data_clean():
    df = T.load_data()
    assert df.text.is_unique and set(df.difficulty) == set(T.LABELS) and (df.question.str.strip() != "").all()


def test_ml_api():
    m = c.get("/api/ml/metrics").get_json(); assert m == json.loads((T.MODEL_DIR / "metrics.json").read_text())
    r = c.post("/api/ml/predict", json={"question": "Explain how gradient descent optimises a neural network loss function"}).get_json()
    assert r["predicted"] in T.LABELS and abs(sum(r["probabilities"].values()) - 1) < 0.01 and r["known_terms"] > 2
    assert c.post("/api/ml/predict", json={"question": "hi"}).status_code == 400
    assert c.post("/api/ml/predict", json={"question": "x" * 1001}).status_code == 400
    assert c.post("/api/ml/predict", data="notjson").status_code == 400
    g = c.post("/api/ml/predict", json={"question": "qwxz zzzzqq plokijuh"}).get_json(); assert g["warning"]


def test_missing_resume_behaviour():
    s = "no-resume"
    r = c.get(f"/api/resume?session_id={s}").get_json(); assert r["resume"] is None and "Upload your resume" in r["message"]
    q = c.get(f"/api/resume/questions?session_id={s}").get_json(); assert q["has_resume"] is False and q["questions"] == []
    n = c.get(f"/api/questions/next?session_id={s}&use_resume=1").get_json()
    assert n["question"] and "No resume uploaded" in n["difficulty_reason"]            # general practice still works


def test_resume_questions_after_upload_and_replace():
    s = "with-resume"
    assert upload(make_pdf(), sid=s).status_code == 200
    q = c.get(f"/api/resume/questions?session_id={s}").get_json(); assert q["has_resume"] and 3 <= len(q["questions"]) <= 8
    topics = {(r["category"], r["topic"]) for r in c.get(f"/api/resume?session_id={s}").get_json()["resume"]["recommended_topics"]}
    assert all((x["category"], x["topic"]) in topics and x["because"] for x in q["questions"])
    assert upload(make_pdf(), "second.pdf", sid=s).get_json()["filename"] == "second.pdf"   # replace
    assert c.get(f"/api/resume?session_id={s}").get_json()["resume"]["filename"] == "second.pdf"
    c.delete(f"/api/resume?session_id={s}"); assert c.get(f"/api/resume/questions?session_id={s}").get_json()["has_resume"] is False


def test_answer_evaluation_correct_partial_incorrect():
    row = A.POOL[A.POOL.answer.str.split().str.len() >= 25].iloc[0]
    ref, words = row.answer, row.answer.split()
    ev = lambda a: A.EVAL.evaluate(row.question, ref, a)
    full, part, off = ev(ref), ev(" ".join(words[: len(words) // 2])), ev("I enjoy football and cooking dinner with friends on weekends")
    assert full["score"] >= 80 and off["score"] <= 25 and off["score"] < part["score"] < full["score"], (full["score"], part["score"], off["score"])
    assert full["missing_concepts"] == [] or len(full["missing_concepts"]) < len(part["missing_concepts"])
    assert part["missing_concepts"] and set(full["breakdown"]) == {"technical_accuracy", "relevance", "completeness"}
    assert ev("ok")["score"] <= 25                                                    # too-short answers are capped


def test_dashboard_has_all_sections_and_updates():
    s = "dash"
    empty = c.get(f"/api/dashboard?session_id={s}").get_json(); assert empty["total"] == 0 and empty["by_difficulty"] == []
    for d in ("Easy", "Hard"):
        q = c.get(f"/api/questions/next?session_id={s}&difficulty={d}&topic=Docker").get_json()["question"]
        ref = A.POOL[A.POOL.uid == q["uid"]].iloc[0].answer
        c.post("/api/evaluate", json={"session_id": s, "uid": q["uid"], "answer": ref})
    j = c.get(f"/api/dashboard?session_id={s}").get_json()
    assert j["total"] == 2 and {x["difficulty"] for x in j["by_difficulty"]} == {"Easy", "Hard"} and len(j["recent"]) == 2
    assert j["strong"] and j["strong"][0]["topic"] == "Docker" and j["weak"] == []


def test_unseen_tracking_and_errors():
    s = "unseen"
    a = c.get(f"/api/questions/next?session_id={s}&topic=Docker&difficulty=Easy").get_json()
    c.post("/api/evaluate", json={"session_id": s, "uid": a["question"]["uid"], "answer": "containers package apps"})
    b = c.get(f"/api/questions/next?session_id={s}&topic=Docker&difficulty=Easy").get_json()
    assert b["remaining"] == a["remaining"] - 1 and b["question"]["uid"] != a["question"]["uid"]
    assert c.get(f"/api/questions/next?session_id={s}&difficulty=Impossible").status_code == 400
    assert c.post("/api/evaluate", data="bad").status_code == 400
    assert c.post("/api/evaluate", json={"session_id": s, "uid": "Q00001", "answer": "x" * 4001}).status_code == 400
    assert c.get("/api/history").status_code == 400 and c.get("/api/nope").status_code == 404


def test_cluster_plot_and_dataset_hub():
    j = c.get("/api/datasets/clusters").get_json(); assert len(j["points"]) == 700 and len(j["clusters"]) == 14
    d = c.get("/api/datasets").get_json(); assert d["preprocessing"] and all(s["top_topics"] for s in d["sources"])
