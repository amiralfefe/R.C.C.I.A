# R.C.C.I.A

R.C.C.I.A est un monorepo portfolio IA/data consacre a des projets de computer vision autour d'images publiques liees au cancer.

Le repo reste strictement educatif : il ne fournit pas de diagnostic medical, ne remplace pas un professionnel de sante et ne doit jamais orienter une decision medicale.

## Project Status

| Projet | Statut | Description |
| --- | --- | --- |
| [Leukemia](projects/leukemia/README.md) | termine | Classification `normal` vs `leukemia_blast`, Streamlit, Grad-CAM, benchmark, error analysis |
| [Lung + Colon](projects/lung_colon/README.md) | V1 reelle | Classification histopathologique LC25000 en 5 classes, ResNet18, Streamlit et Grad-CAM |
| Breast | prevu | Projet futur autour d'un dataset public type tumeurs benignes/malignes |
| Metastasis | prevu | Projet futur oriente detection/patch classification |
| MultiCancer | prevu | Synthese multi-projets et comparaison transversale |

## Projet Principal Termine

Le sous-projet leucemie est la premiere brique complete du portfolio :

- V1 : pipeline ML reel, Streamlit et Grad-CAM ;
- V2 : comparaison de modeles ;
- V2.1 : benchmark reel ResNet18 / MobileNetV3 / EfficientNet ;
- V2.2 : analyse des erreurs et interpretabilite ;
- V2.3 : pack CV, LinkedIn, entretien et release notes.

Tag de reference :

```text
v2.2-error-analysis
```

## Structure

```text
R.C.C.I.A/
|-- README.md
|-- requirements.txt
|-- requirements-dev.txt
|-- pyproject.toml
|-- docs/
|   |-- ROADMAP.md
|   |-- PORTFOLIO_OVERVIEW.md
|   `-- assets/
`-- projects/
    |-- leukemia/
    |   |-- README.md
    |   |-- app.py
    |   |-- rccia_leukemia/
    |   |-- scripts/
    |   |-- tests/
    |   |-- docs/
    |   |-- data/
    |   `-- outputs/
    `-- lung_colon/
        |-- README.md
        |-- app.py
        |-- rccia_lung_colon/
        |-- scripts/
        |-- tests/
        |-- docs/
        |-- data/
        `-- outputs/
```

## Installation Globale

Depuis la racine :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Lancer Les Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Lancer Le Projet Leucemie

```powershell
cd projects\leukemia
..\..\.venv\Scripts\streamlit.exe run app.py
```

## Lancer Le Projet Lung + Colon

```powershell
cd projects\lung_colon
..\..\.venv\Scripts\streamlit.exe run app.py
```

V1 reelle : baseline ResNet18 pre-entrainee sur LC25000, 25 000 images, 5 classes, accuracy test **0.9965**. Voir [projects/lung_colon/README.md](projects/lung_colon/README.md) pour le protocole et les limites.

Les donnees, checkpoints, outputs, tokens Kaggle et environnements virtuels ne sont pas versionnes.

## Docs Globales

- [Roadmap](docs/ROADMAP.md)
- [Portfolio overview](docs/PORTFOLIO_OVERVIEW.md)
- [Leukemia project summary](projects/leukemia/docs/PROJECT_SUMMARY.md)
- [Leukemia interview pitch](projects/leukemia/docs/INTERVIEW_PITCH.md)
- [Leukemia LinkedIn draft](projects/leukemia/docs/LINKEDIN_DRAFT.md)
