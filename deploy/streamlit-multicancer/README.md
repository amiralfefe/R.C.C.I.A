# MultiCancer Streamlit Community Cloud Deployment

This directory contains only the packaging layer used to run the frozen MultiCancer V1
hub on Streamlit Community Cloud.

## Architecture

```text
Streamlit Community Cloud
  -> deployment bootstrap
  -> private Hugging Face model repository
  -> five checkpoints on ephemeral disk
  -> projects/multicancer/app.py
  -> existing one-model-at-a-time lazy loading
```

The bootstrap imports the existing D0/D1 downloader from
`deploy/hf-multicancer/download_models.py`. It does not duplicate the remote-to-local
checkpoint mapping. On each Streamlit rerun, it first checks the five local files and
contacts Hugging Face only when at least one checkpoint is missing.

No PyTorch model is loaded by the bootstrap. The existing `ModelManager` remains solely
responsible for loading the explicitly selected specialized pipeline and unloading the
previous one.

## Data And Security

- no dataset is downloaded;
- no checkpoint is stored in GitHub;
- no token or `secrets.toml` file is versioned;
- no user image is intentionally stored after the Streamlit session;
- the Hugging Face repository remains private;
- production access requires a dedicated read-only `HF_TOKEN`.

This application is an educational AI/data portfolio demonstration only. It is not a
medical device, diagnostic tool, clinical validation or source of health advice.

See [STREAMLIT_SETUP.md](STREAMLIT_SETUP.md) for the manual deployment coordinates and
[DEPLOYMENT_VALIDATION.md](DEPLOYMENT_VALIDATION.md) for the factual validation state.
