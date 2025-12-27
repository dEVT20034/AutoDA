# AutoDA Automation Hub

AutoDA is a full-stack automation studio built with Flask that lets analysts ingest structured datasets, scrape the web, automate NLP pipelines, prepare image datasets, and export explainable reports. The platform bundles Auto Mode orchestration, manual controls for every pipeline step, authentication, and a polished landing/workflow experience.

---

## Table of contents
1. [Key features](#key-features)
2. [Tech stack & tooling](#tech-stack--tooling)
3. [Project structure](#project-structure)
4. [Workflows in detail](#workflows-in-detail)
5. [Setting up the project](#setting-up-the-project)
6. [Running the app & tests](#running-the-app--tests)
7. [Configuration & API keys](#configuration--api-keys)
8. [Contact / support form](#contact--support-form)

---

## Key features

| Area | Description |
| --- | --- |
| Landing & Workflow Hub | Hero CTAs, animated stats, workflow guide, personalized greeting if logged in, integrated contact form that emails `devt@gmail.com`. |
| Structured Data Pipeline | Upload CSV/TSV/Excel/Parquet, profile, clean, engineer features, split, train & compare models, export cleaned data + reports. |
| Auto Mode | One-click “ingestion → reports” pipeline with audit trail, artifact downloads, metrics and error handling. |
| Manual Phases | Dedicated pages for Profiling, Cleaning, Feature Engineering, Feature Selection, Split & Validation, Training & Best-fit selection. |
| Feature Engineering/Selection | Manual toggles, auto-select helpers, dropdown-driven configurations when Auto Mode is skipped. |
| Split & Validation | Configurable strategies (hold-out, stratified, time-series, k-fold), previews of train/test rows, leakage warnings. |
| Training & Best Fit | Task detection (regression/classification/clustering), leaderboard, winner explanation, error analysis visuals, artifact downloads. |
| Visualization Hub | Auto-generated Plotly charts, filters, correlation heatmaps, dataset download per chart. |
| Reports & Artifacts | Executive summaries, reproducibility appendix, export of cleaned data/config/report PDF or HTML. |
| Prediction Sandbox | Upload/paste new rows, schema validation, run predictions, download batch outputs, drift guardrails. |
| Web Scraper | URL + goal + optional headers ➜ BeautifulSoup extraction + Gemini-generated dataset, download button in Prototype step. |
| Image Processing Hub | Drag/drop queue, 50+ preprocessing toggles, Auto Mode dataset builder, preview metrics (PSNR/SSIM), download button under preview. |
| Contact Form | Styled form (name/email/subject/message) posts to backend, emails `devt@gmail.com` via SendGrid-compatible API key, prefills user info when logged in. |

---

## Tech stack & tooling

### Backend
- **Flask** routing, session management, templating.
- **SQLAlchemy + MySQL** for users and project metadata.
- **Pandas, NumPy** for ingestion, cleaning, engineering.
- **scikit-learn** for preprocessing, model training, metrics, Auto Mode orchestration.
- **BeautifulSoup & requests** for web scraping.
- **Plotly** for interactive charts.
- **SendGrid-compatible API** (via `CONTACT_EMAIL_API_KEY`) to deliver contact form messages.
- **Google Gemini API** (via `GEMINI_API_KEY`) to generate structured web-scrape datasets.

### Frontend
- Jinja2 templates layered on `base.html`.
- Vanilla JS (`static/js/main.js`, `static/js/image_hub.js`) for dropdowns, auto-dismiss flashes, drag/drop, preview controls.
- CSS (`static/css/main.css`, `landing.css`, `image_hub.css`) for the dark theme, responsive layout, hero cards, contact form, and image hub styling.

### Tooling
- `python -m compileall app` for quick syntax verification.
- `tests/run_auto_test.py` smoke tests.

---

## Project structure

```
cap/
├── app/
│   ├── __init__.py          # Flask factory + DB bootstrap
│   ├── routes.py            # Main blueprint, contact handler, web scraper, image hub
│   ├── context.py           # Page metadata & breadcrumb builders
│   ├── image_hub.py         # Image processing helpers, auto dataset builder
│   ├── nlp_automation.py    # NLP blueprint routes
│   ├── state.py             # Project store, audit helpers
│   └── models.py            # SQLAlchemy models (ProjectMeta, User, etc.)
├── templates/
│   ├── base.html, auth_base.html
│   ├── landing.html, workflow_select.html (public pages)
│   ├── [pipeline].html      # ingestion, profiling, cleaning, feature_* pages, etc.
│   ├── web_scraper.html, image_hub.html
│   └── signin.html, signup.html
├── static/
│   ├── css/ (main.css, landing.css, image_hub.css)
│   ├── js/  (main.js, image_hub.js)
│   └── assets/
├── uploads/                 # cleaned datasets, reports, web-scrape CSVs, image ZIPs
└── README.md
```

---

## Workflows in detail

### Structured Data Pipeline
1. **Data Ingestion** – multi-format upload, DB connectors, schema sniffing, duplicate guard.
2. **Profiling** – statistics, semantic detection, correlation heatmaps, missingness matrix.
3. **Cleaning** – missing-value policies (numeric/categorical/date), dedupe, outlier handling, text normalization, leakage guardrails.
4. **Feature Engineering** – numeric transforms, date expansion, categorical encoding, text vectorization, modality-specific prep, manual toggles.
5. **Feature Selection** – filter/wrapper/embedded selection, versioned feature sets, auto-select button.
6. **Split & Validation** – hold-out / k-fold / stratified / time-series options, sample previews, class balance charts.
7. **Training & Best Fit** – auto task detection, candidate leaderboard, winner justification, PSNR/SSIM metrics where applicable, artifact downloads.
8. **Visualization & Reports** – Plotly charts, EDA summary, residuals/confusion matrix, report export (PDF/HTML), artifact hub.

### NLP Workflow
- Dedicated blueprint handles text ingestion, cleaning, tokenization, embeddings/vectorization, sentiment, clustering, ready for Auto Mode or manual training.

### Web Scraper
1. User enters URL, goal/intention, optional API header/token.
2. Backend fetches page via `requests`, scrapes copy/headings with BeautifulSoup.
3. Gemini API builds a structured dataset (headline/detail/category). Fallback uses local extraction.
4. CSV saved under `uploads/web_scrapes/`, download button appears in Prototype step.

### Image Processing Hub
- Drag/drop queue, queue cards, selection toggles.
- Configurable transformations: resize, crop/pad, rotation, color adjustments, filtering, augmentation.
- Manual apply vs. Auto Mode dataset builder (prompts for size/preset, generates ZIP).
- Preview metrics (PSNR, SSIM) plus download button relocated below “Preview & metrics”.

### Contact / Support
- Landing + workflow pages display a two-column contact panel.
- Form posts to `/contact`, server sends email to `devt@gmail.com`.
- Prefills name/email when authenticated and offers dashboard/project shortcuts.

---

## Setting up the project

1. **Clone & install**
   ```bash
   git clone <repo>
   cd cap
   python -m venv .venv
   .\.venv\Scripts\activate  # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Database**
   - Update `MYSQL_SETTINGS` in `app/__init__.py` or export environment variables.
   - The app auto-creates the DB (via `ensure_mysql_database()`).

3. **Environment variables**
   ```bash
   set GEMINI_API_KEY=your_gemini_key
   set CONTACT_EMAIL_API_KEY=your_sendgrid_key
   # optional overrides: FLASK_SECRET_KEY, SQLALCHEMY_DATABASE_URI, etc.
   ```

4. **Run the app**
   ```bash
   flask --app app:create_app run --debug
   ```
   or
   ```bash
   python app.py
   ```

---

## Running the app & tests

| Command | Purpose |
| --- | --- |
| `flask --app app:create_app run --debug` | Start dev server with hot reload. |
| `python -m compileall app` | Quick syntax check across the project. |
| `python tests/run_auto_test.py` | Smoke test Auto Mode pipeline. |
| Add your own `pytest` / `ruff` / `mypy` integrations as needed. |

---

## Configuration & API keys

| Setting | Description |
| --- | --- |
| `GEMINI_API_KEY` | Required for the web scraper’s structured dataset generation. |
| `CONTACT_EMAIL_API_KEY` | SendGrid (or compatible) API key for the contact form. |
| `SQLALCHEMY_DATABASE_URI` | Automatically assembled from `MYSQL_SETTINGS`, can be overridden. |
| `UPLOAD_FOLDER` | Defaults to `<repo>/uploads`; stores cleaned data, reports, image ZIPs, scraper CSVs. |

Ensure `uploads/` is writable; Auto Mode and the scraper rely on it for artifact persistence.

---

## Contact / support form

- **Location:** bottom of `landing.html` and `workflow_select.html`.
- **Fields:** name, email, subject, message (email + message required).
- **Behavior:** POST `/contact` ➜ validates ➜ sends email to `devt@gmail.com` ➜ flashes success/error (auto-dismiss after 5s).
- **Prefill:** uses `session.user_name` and `session.user_email` to prefill fields when authenticated.
- **CTAs:** logged-in users see “Go to dashboard / View projects”; guests see “Sign up now / I’m already a member”.

---

## Need help?

- Use the in-app contact form or email `devt28173@gmail.com`.
- Contributions welcome—open issues or PRs for bugs, features, or UX polish.
