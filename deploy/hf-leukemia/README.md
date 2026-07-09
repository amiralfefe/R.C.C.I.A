---
title: RCCIA Leukemia Demo
sdk: streamlit
app_file: app.py
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
For a public demo, place the checkpoint in the Hugging Face Space repository at:

```text
outputs/best_model.pt
```

or keep the monorepo layout in the Space and place it at:

```text
projects/leukemia/outputs/best_model.pt
```

Use the Space repository, Git LFS, or the Hugging Face web upload for the model
artifact. Do not add the checkpoint to the main GitHub repo unless that is a
deliberate release decision.

## Suggested Space structure

Minimal Space repository:

```text
app.py
rccia_leukemia/
outputs/
  best_model.pt
requirements.txt
README.md
```

Files to copy from the main repo:

```text
projects/leukemia/app.py -> app.py
projects/leukemia/rccia_leukemia/ -> rccia_leukemia/
deploy/hf-leukemia/requirements.txt -> requirements.txt
deploy/hf-leukemia/README.md -> README.md
```

Alternative monorepo-style Space:

```text
deploy/hf-leukemia/app.py
projects/leukemia/app.py
projects/leukemia/rccia_leukemia/
projects/leukemia/outputs/best_model.pt
requirements.txt
```

In that case, set the Space `app_file` metadata to:

```text
deploy/hf-leukemia/app.py
```

## Cloud requirements

The deployment requirements use `opencv-python-headless` instead of
`opencv-python`, because the Streamlit demo does not need OpenCV desktop UI
bindings in the cloud.

## Next deployment steps

1. Create a new Hugging Face Space with SDK `streamlit`.
2. Copy the app code and `rccia_leukemia` package into the Space.
3. Upload `best_model.pt` into `outputs/`.
4. Add `requirements.txt`.
5. Launch the Space and confirm the checkpoint is loaded.
6. Add the final Space URL to the portfolio Demo button.
