"""Build the unified open-ended Q&A dataset from the two raw CSV files.

Rules (nothing is invented):
  * MCQ -> open-ended: the reference answer is the text of the option the dataset marks correct.
  * Aptitude sections (no worked solution in the data), option-dependent stems, and rows whose
    answer key breaks the dataset's own pattern are flagged `needs_review`, not fabricated.
"""
import json, re, unicodedata
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
APTITUDE = {"Numerical Ability", "Logical Reasoning", "Verbal Ability"}
OPTION_DEP = re.compile(r"following|spot the error|odd one out|choose|select the|which of these|"
                        r"\bnot\b|\bexcept\b|both a and|\(a\)|identify the", re.I)
DIFF = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
CAT_FIX = {"General Program": "General Programming", "Database Systems": "Database and SQL"}


def read_csv(path):
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError:
            continue
    raise ValueError(path)


def clean(x):
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKC", str(x)).replace("\ufffd", "")
    return re.sub(r"\s+", " ", x).strip()


def norm_q(q):
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def convert_software(df):
    rows = []
    for _, r in df.iterrows():
        q, a = clean(r["Question"]), clean(r["Answer"])
        cat = CAT_FIX.get(clean(r["Category"]), clean(r["Category"]))
        status, note = ("invalid", "missing question or answer") if not q or not a else ("converted", "")
        rows.append(dict(source="Software_Questions", src_id=f"SW_{int(r['Question Number']):03d}",
                         category="Software Interview Bank", topic=cat or "General",
                         difficulty=DIFF.get(clean(r["Difficulty"]).lower(), "Medium"),
                         question=q, answer=a, status=status, note=note))
    return rows


def convert_mcq(df):
    rows = []
    for _, r in df.iterrows():
        q, sec = clean(r["question"]), clean(r["section"])
        key = clean(r["answer"]).upper()
        opts = {k: clean(r[k]) for k in "ABCD"}
        status, note, ans = "converted", "", opts.get(key, "")
        if not q or key not in opts or not ans:
            status, note = "invalid", "missing question, invalid answer letter, or empty correct option"
        elif sec in APTITUDE:
            status, note = "needs_review", "Aptitude item: data holds only the final answer, no worked explanation to derive one from"
        elif OPTION_DEP.search(q):
            status, note = "needs_review", "Question depends on the answer options, so it is not meaningful as open-ended"
        elif key != "A":
            status, note = "needs_review", "Answer key differs from this dataset's usual pattern (97% are 'A'); verify the key"
        question, answer = q, ans
        if status == "converted":
            if re.search(r"(:|_{2,}|\.\.\.)\s*$", q):
                stem = q.rstrip(":").strip()
                question, answer = f"Complete the statement and explain: {stem}", f"{stem} {ans}."
            else:
                answer = ans if ans.endswith(".") else ans + "."
        rows.append(dict(source="Placement_Interview_Quiz_Questions", src_id=clean(r["id"]), category=sec,
                         topic=clean(r["topic"]), difficulty=DIFF.get(clean(r["difficulty"]).lower(), "Medium"),
                         question=question, answer=answer, status=status, note=note))
    return rows


def build():
    sw, enc1 = read_csv(RAW / "Software_Questions.csv")
    mcq, enc2 = read_csv(RAW / "Placement_Interview_Quiz_Questions.csv")
    rows = convert_software(sw) + convert_mcq(mcq)
    seen, dup = {}, 0
    for r in rows:
        if r["status"] == "invalid":
            continue
        k = norm_q(r["question"])
        if k in seen:
            r["status"], r["note"] = "duplicate", f"Duplicate of {seen[k]}"
            dup += 1
        else:
            seen[k] = r["src_id"]
    df = pd.DataFrame(rows)
    df.insert(0, "uid", [f"Q{i:05d}" for i in range(1, len(df) + 1)])
    from nlp import add_clusters
    df = add_clusters(df)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "unified_questions.csv", index=False)
    report = dict(files={"Software_Questions": dict(rows=len(sw), columns=list(sw.columns), encoding=enc1),
                         "Placement_Interview_Quiz_Questions": dict(rows=len(mcq), columns=list(mcq.columns), encoding=enc2)},
                  quality=dict(duplicate_ids_in_mcq=int(mcq["id"].duplicated().sum()), duplicate_questions_flagged=dup,
                               missing_option_cells=int(mcq[list("ABCD")].isna().sum().sum()),
                               answer_key_distribution=mcq["answer"].value_counts().to_dict()))
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2))
    return df, report


if __name__ == "__main__":
    df, rep = build()
    print(df.groupby(["source", "status"]).size())
    print(json.dumps(rep["quality"]))
    print(sorted(df[df.status == "converted"].topic.unique()))
