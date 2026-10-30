# Interview Twin - AI/ML Interview Preparation Assistant

Flask + SQLite + scikit-learn + plain HTML/CSS/JavaScript. Runs fully on your own computer, no paid services.

## Run it in VS Code on Windows
1. Install **Python 3.10 or newer** from python.org (tick *Add Python to PATH*) and **VS Code** with the *Python* and *Jupyter* extensions.
2. Unzip the project, then in VS Code use **File > Open Folder** and choose the `interview-twin` folder.
3. Open a terminal with **Ctrl + `** (PowerShell) and run:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   (If activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first, then activate again.)
4. Start the website:
   ```
   cd backend
   python app.py
   ```
   Open **http://127.0.0.1:5000** in your browser. Stop the server with Ctrl+C.
5. Run the tests (from the project folder, not `backend`): `python -m pytest tests -q`
6. Train the model yourself: open `notebooks/difficulty_classifier.ipynb`, choose the `.venv` kernel (top right), click **Run All**.

`models/` already contains the trained model. If those files are ever missing, the app trains them automatically on startup.

## Folder structure
```
backend/app.py          Flask routes (API)           backend/db.py        SQLite tables and helpers
backend/nlp.py          answer evaluation, K-Means   backend/resume.py    PDF text, skill dictionary, topic mapping
backend/data_prep.py    builds the clean dataset     backend/report.py    PDF report
backend/train_model.py  same training as notebook    frontend/index.html  the whole website
notebooks/              explained ML training        models/              saved model, vectorizer, metrics.json
data/raw/               your 2 original CSVs         data/processed/      cleaned unified dataset
tests/                  automated tests
```

## How the main parts work (beginner-friendly)
- **Dataset** (`data_prep.py`): both CSVs are merged. MCQs become open-ended questions whose reference answer is the option the dataset marks correct (nothing is invented). Duplicates, aptitude items and suspicious rows are flagged and left out (3,700 usable questions).
- **Difficulty classifier** (`notebooks/`, `train_model.py`): **TF-IDF** turns each question into numbers (rare, meaningful words weigh more); **Logistic Regression** learns which words go with Easy/Medium/Hard. It is tested on 20% of questions it never saw. The website only *loads* the saved files (`joblib`); it never retrains per request.
- **Answer evaluation** (`nlp.py`): compares your answer with the reference using TF-IDF **cosine similarity** (how similar two texts are), key-concept coverage (which important words you used or missed) and shared two-word phrases. It gives accuracy-estimate, relevance, completeness and missing concepts.
- **Recommendations** (`/api/recommend`): unseen questions only; weak topics and resume topics rank higher; questions worded like ones you scored low on rank higher (cosine similarity); near-duplicates are skipped. With no history it says the picks are random.
- **Adaptive difficulty**: average of your last 3 scores; 70+ goes up, 40 or below goes down, otherwise stays.
- **Resume**: `pdfplumber` extracts text, a skill dictionary (`SKILL_TABLE` in `resume.py`) finds skills, each skill maps to real (category, topic) pairs in the dataset. You can remove or add skills and topics update.
- **Accounts**: register/login with hashed passwords (Werkzeug). Guest mode still works; guest history moves into your account when you register or sign in.

## Honest limitations
- Difficulty model: ~53% accuracy vs 44.5% for always guessing the commonest label (macro-F1 0.50 vs 0.21). Better than chance, not reliable. See the ML Model page.
- Answer scoring is **lexical, not semantic**: it cannot confirm an answer is technically correct, and a correct paraphrase can score low.
- The dataset has **no real Java, C/C++ or Operating Systems questions**, and very few Networking/OOP/Software-Engineering ones, so those topics cannot be practised properly until questions are added.
- Scanned (image-only) PDFs are not supported: there is no OCR.
- Prototype security: no login rate limiting, no CSRF tokens, no email/password reset. Set the `SECRET_KEY`-style file `data/.secret_key` private; do not expose this server to the internet as is.
