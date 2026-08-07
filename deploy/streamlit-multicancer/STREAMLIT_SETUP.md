# Streamlit Community Cloud Setup

## Deployment Coordinates

- GitHub repository: `amiralfefe/R.C.C.I.A`
- Branch: `deploy/multicancer-v1-streamlit`
- Entrypoint: `deploy/streamlit-multicancer/app.py`
- Requested app subdomain: `rccia-multicancer`
- Local validated Python version: `3.11.8`

Select Python **3.11** in Advanced settings. This is the closest Community Cloud choice
to the local runtime and must be chosen before the first deployment.

## Secrets

Add these root-level values directly in the Streamlit Community Cloud **Secrets** field:

```toml
HF_MODEL_REPO_ID = "amiralfefe/rccia-multicancer-models"
HF_TOKEN = "<READ_ONLY_TOKEN>"
```

Root-level Streamlit secrets are exposed as environment variables, so the existing
downloader can keep reading `os.environ`. Create a dedicated Hugging Face read-only token
and paste it directly into Streamlit. Never send it through chat, commit it, place it in
a screenshot or save it in a versioned `secrets.toml` file.

## Create The App

1. Open `https://share.streamlit.io` and select **Create app**.
2. Choose repository `amiralfefe/R.C.C.I.A`.
3. Choose branch `deploy/multicancer-v1-streamlit`.
4. Set the main file path to `deploy/streamlit-multicancer/app.py`.
5. Open **Advanced settings**.
6. Select Python `3.11`.
7. Paste the two root-level secrets above with the real read-only token.
8. Deploy and retain the resulting public `streamlit.app` URL.

Community Cloud executes the app from the repository root. The dedicated requirements
file beside the entrypoint pins the locally validated inference dependencies.

## Startup And Resources

The bootstrap downloads approximately 132 MiB of checkpoints when the instance disk
does not already contain them. No startup-time promise is made before public measurement.
Community Cloud can hibernate inactive apps, and an instance wake-up or replacement may
require another download because local storage is not guaranteed to persist.

Community Cloud resources are limited. The hub therefore preserves its existing lifecycle:
checkpoints may coexist on disk, but only the explicitly selected model is loaded in RAM.
Switching project or LungColon mode unloads the active model before another is loaded.

## Public Validation

After the URL exists, validate cold start, the five paths, Grad-CAM, Metastasis threshold
exploration, project/mode unload behavior, degraded states and a mobile viewport. Record
only observed results in `DEPLOYMENT_VALIDATION.md`.
