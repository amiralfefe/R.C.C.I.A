# R.C.C.I.A

R.C.C.I.A est un monorepo portfolio IA/data consacre a des projets de computer vision autour d'images publiques liees au cancer.

Le repo reste strictement educatif : il ne fournit pas de diagnostic medical, ne remplace pas un professionnel de sante et ne doit jamais orienter une decision medicale.

## Project Status

| Projet | Statut | Description |
| --- | --- | --- |
| [Leukemia](projects/leukemia/README.md) | termine | Classification `normal` vs `leukemia_blast`, Streamlit, Grad-CAM, benchmark, error analysis |
| [Lung + Colon](projects/lung_colon/README.md) | V2.2 complete + final pack | Classification LC25000 en 5 classes, Streamlit, Grad-CAM, benchmark multi-modeles, analyse des erreurs, mode binaire benin/malin et docs portfolio |
| [Breast](projects/breast/README.md) | V2 model comparison | Classification BreakHis benign/malignant avec split patient-aware, Streamlit et benchmark ResNet18 / MobileNetV3 / EfficientNet |
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
    |-- lung_colon/
    |   |-- README.md
    |   |-- app.py
    |   |-- rccia_lung_colon/
    |   |-- scripts/
    |   |-- tests/
    |   |-- docs/
    |   |-- data/
    |   `-- outputs/
    `-- breast/
        |-- README.md
        |-- app.py
        |-- rccia_breast/
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

V1 reelle : baseline ResNet18 pre-entrainee sur LC25000, 25 000 images, 5 classes, accuracy test **0.9965**.

V2 : benchmark comparatif ResNet18 / MobileNetV3 small / EfficientNet-B0. Meilleur score observe : EfficientNet-B0 avec **0.9992** accuracy test.

V2.1 : analyse des erreurs sur EfficientNet-B0, **3 erreurs sur 3 750 images test**, toutes entre `lung_adenocarcinoma` et `lung_squamous_cell_carcinoma`.

V2.2 : mode binaire `benign` vs `malignant` avec ResNet18 pre-entraine, accuracy test **1.0000** et recall `malignant` **1.0000** sur le split local binaire. Ces resultats sont a interpreter prudemment car LC25000 est un benchmark public relativement facile. Voir [projects/lung_colon/README.md](projects/lung_colon/README.md) pour le protocole et les limites.

Pack portfolio Lung + Colon :

- [Project summary](projects/lung_colon/docs/PROJECT_SUMMARY.md)
- [Interview pitch](projects/lung_colon/docs/INTERVIEW_PITCH.md)
- [LinkedIn draft](projects/lung_colon/docs/LINKEDIN_DRAFT.md)
- [Release notes V2.2](projects/lung_colon/docs/RELEASE_NOTES_V2_2.md)

Les donnees, checkpoints, outputs, tokens Kaggle et environnements virtuels ne sont pas versionnes.

## Lancer Le Projet Breast

```powershell
cd projects\breast
..\..\.venv\Scripts\streamlit.exe run app.py
```

V1 reelle : dataset BreakHis Kaggle, 7 909 images, 81 patients detectes, split patient-aware sans overlap patient, baseline ResNet18 pre-entrainee, accuracy test **0.8947**. Le recall `malignant` est **0.9466** sur le split local patient-aware.

V2 : benchmark patient-aware ResNet18 / MobileNetV3 small / EfficientNet-B0. Meilleur score observe : EfficientNet-B0 avec **0.9122** accuracy test et **0.9004** macro F1. MobileNetV3 obtient le meilleur recall `malignant` (**0.9979**) et le temps d'entrainement le plus court, mais avec un F1 `benign` plus faible. Ces resultats restent experimentaux et ne constituent pas une validation clinique.

## Docs Globales

- [Roadmap](docs/ROADMAP.md)
- [Portfolio overview](docs/PORTFOLIO_OVERVIEW.md)
- [Leukemia project summary](projects/leukemia/docs/PROJECT_SUMMARY.md)
- [Leukemia interview pitch](projects/leukemia/docs/INTERVIEW_PITCH.md)
- [Leukemia LinkedIn draft](projects/leukemia/docs/LINKEDIN_DRAFT.md)
- [Breast V2 model comparison](projects/breast/docs/V2_MODEL_COMPARISON.md)
