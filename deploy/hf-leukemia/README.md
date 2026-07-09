---
title: R.C.C.I.A Leukemia Demo
emoji: 🩸
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# R.C.C.I.A Leukemia - Hugging Face Space

This folder documents a deployment target for the Leukemia Streamlit demo.

The demo is educational and portfolio-oriented only. It is not a medical device,
does not provide a diagnosis, and must not guide a health decision.

## Local app

Main app:

```powershell
cd C:\VSCODE\R.C.C.I.A\projects\leukemia
..\..\.venv\Scripts\python.exe -m streamlit run app.py
```

The app resolves relative artifacts from `projects/leukemia`, so the default
checkpoint path is:

```text
projects/leukemia/outputs/best_model.pt
```

## Checkpoint policy

`best_model.pt` is intentionally not committed to the main GitHub repository.
For the Docker Space structure documented below, place the checkpoint in the
Hugging Face Space repository at:

```text
projects/leukemia/outputs/best_model.pt
```

If you later create a minimal Space where `projects/leukemia/app.py` is copied
to the Space root as `app.py`, then the equivalent checkpoint path becomes:

```text
outputs/best_model.pt
```

Use the Space repository, Git LFS, or the Hugging Face web upload for the model
artifact. Do not add the checkpoint to the main GitHub repo unless that is a
deliberate release decision.

## Required Space structure

Use a dedicated Docker Space repository with this structure:

```text
Dockerfile
app.py
requirements.txt
README.md
projects/
  leukemia/
    app.py
    rccia_leukemia/
    outputs/
      best_model.pt
```

Files to copy from the main repo:

```text
deploy/hf-leukemia/Dockerfile -> Dockerfile
deploy/hf-leukemia/app.py -> app.py
deploy/hf-leukemia/requirements.txt -> requirements.txt
deploy/hf-leukemia/README.md -> README.md
projects/leukemia/app.py -> projects/leukemia/app.py
projects/leukemia/rccia_leukemia/ -> projects/leukemia/rccia_leukemia/
```

## Cloud requirements

The deployment requirements use `opencv-python-headless` instead of
`opencv-python`, because the Streamlit demo does not need OpenCV desktop UI
bindings in the cloud.

## Docker runtime

The Dockerfile:

- uses `python:3.11-slim`;
- installs `requirements.txt`;
- copies the Space repository into `/home/user/app`;
- starts Streamlit on `0.0.0.0:7860`.

Hugging Face Docker Spaces expose the port declared by `app_port` in this
README metadata.

## Next deployment steps

1. Create a new Hugging Face Space with SDK `Docker`.
2. Copy the files listed in the required Space structure.
3. Upload `best_model.pt` into `projects/leukemia/outputs/`.
4. Launch the Space and confirm the checkpoint is loaded.
5. Add the final Space URL to the portfolio Demo button.
