# SADAT — Premium Student Portal

Student-only Streamlit application using the supplied trained `classifier.joblib` and `compression.joblib` artifacts, with a real review → confirm → save → track → submission-experience feedback flow.

## Run

Use Python 3.11 or 3.12 and install dependencies:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The repository root must contain `app.py`, both `.joblib` artifacts, `logo.jpg`, `requirements.txt`, and `assets/theme.css`. Streamlit config belongs in `.streamlit/config.toml`.

## Model behavior

The classifier is a research baseline with four labels (`CARDS`, `EXAMS`, `RECORDS`, `RESULTS`) mapped to **proposed** offices. The compression model chooses whether to apply conservative deterministic extraction rules. It is not a generative summarizer. The classifier pickle requires `normalize_arabic` to be defined in `__main__`; the adapter registers the original callable before loading trusted, local artifacts. **Never load untrusted joblib files.**

## Data protection and deployment

SQLite is local and ephemeral on Streamlit Community Cloud. This application does **not** provide institutional identity verification, durable persistence, staff workflow, or confirmed resolution. It must **not** be used for real identifiable complaints on a public deployment until a durable database, authentication, retention controls, access restrictions, and privacy/security review are implemented. Keep secrets and database files out of Git. The reference and private tracking code are both required for case retrieval.

## Testing

Run `python -m pytest -q` and `python -m py_compile app.py`. Use browser-based manual QA at 390, 768, and 1440 pixels for English and Arabic before release. A Streamlit browser session is not equivalent to automated mobile verification.
