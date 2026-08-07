---
title: RCCIA MultiCancer
sdk: docker
app_port: 7860
pinned: false
---

# R.C.C.I.A MultiCancer

Hub Streamlit public regroupant cinq parcours specialises de classification d'images :
Leukemia, Breast, Metastasis, LungColon multiclass et LungColon binary.

Le conteneur recupere le code fige au tag Git `multicancer-v1`, telecharge les cinq
checkpoints depuis un repository Hugging Face Model distinct, puis lance Streamlit sur
le port `7860`. Les modeles restent charges a la demande et un seul modele est conserve
en memoire par le `ModelManager` existant.

## Configuration Du Space

- Variable `HF_MODEL_REPO_ID` : identifiant du repository contenant les checkpoints.
- Secret `HF_TOKEN` : token Hugging Face en lecture seule, requis si le repository de
  modeles est prive.

Les checkpoints, datasets, outputs et secrets ne sont pas inclus dans ce repository.
Consulter [SPACE_SETUP.md](SPACE_SETUP.md) et
[MODEL_REPO_LAYOUT.md](MODEL_REPO_LAYOUT.md) avant la publication.

## Cadre D'Utilisation

Ce Space est un demonstrateur educatif et portfolio. Il ne constitue pas un dispositif
medical, un outil de diagnostic, une validation clinique ou une aide a la decision de
sante. Les probabilites, Grad-CAM et variations de seuil sont uniquement exploratoires.
