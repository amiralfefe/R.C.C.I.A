# Leukemia Streamlit Community Cloud Deployment

This directory packages the existing Leukemia Streamlit demo for Community Cloud.
It does not contain a model checkpoint and does not change the frozen inference code.

## Checkpoint Flow

The bootstrap reuses `deploy/hf-multicancer/download_models.py` and its central
remote-to-local mapping. It first checks:

`projects/leukemia/outputs/best_model.pt`

When that ignored local file is absent or empty, it downloads only
`leukemia/best_model.pt` from the private Hugging Face model repository. The downloaded
file is installed at the same local path expected by the existing app. Streamlit reruns
reuse that file, while `projects/leukemia/app.py` keeps the loaded model in
`st.cache_resource`.

## Local Launch

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run deploy\streamlit-leukemia\app.py
```

With the local checkpoint present, no Hugging Face configuration is required.

## Community Cloud

- Repository: `amiralfefe/R.C.C.I.A`
- Branch: `main`
- Main file path: `deploy/streamlit-leukemia/app.py`
- Python: `3.11`

Configure these root-level values in the Streamlit Community Cloud Secrets field:

```toml
HF_MODEL_REPO_ID = "amiralfefe/rccia-multicancer-models"
HF_TOKEN = "<READ_ONLY_TOKEN>"
```

Use a dedicated Hugging Face read-only token with access to the private model repository.
Never commit the token, a `.streamlit/secrets.toml` file or the checkpoint.

This application is an educational AI/data portfolio demonstration only. It is not a
medical device, diagnostic tool, clinical validation or source of health advice.
