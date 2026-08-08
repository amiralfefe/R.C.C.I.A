# R.C.C.I.A

R.C.C.I.A est un monorepo portfolio IA/data consacre a des projets de computer vision autour d'images publiques liees au cancer.

Le repo reste strictement educatif : il ne fournit pas de diagnostic medical, ne remplace pas un professionnel de sante et ne doit jamais orienter une decision medicale.

**Demo publique MultiCancer :** https://rccia-multicancer.streamlit.app/

La demo regroupe les cinq parcours specialises avec chargement paresseux, probabilites
et Grad-CAM. Elle reste un demonstrateur educatif / portfolio sans validation clinique.

## Project Status

| Projet | Statut | Description |
| --- | --- | --- |
| [Leukemia](projects/leukemia/README.md) | termine | Classification `normal` vs `leukemia_blast`, Streamlit, Grad-CAM, benchmark, error analysis |
| [Lung + Colon](projects/lung_colon/README.md) | V2.2 complete + final pack | Classification LC25000 en 5 classes, Streamlit, Grad-CAM, benchmark multi-modeles, analyse des erreurs, mode binaire benin/malin et docs portfolio |
| [Breast](projects/breast/README.md) | V2.2 complete + final pack | Classification BreakHis benign/malignant avec split patient-aware, Streamlit, benchmark ResNet18 / MobileNetV3 / EfficientNet, analyse d'erreurs et docs portfolio |
| [Metastasis](projects/metastasis/README.md) | V2.2 complete + final pack | Classification de patches `non_metastatic` vs `metastatic`, pipeline PyTorch, ROC-AUC / PR-AUC, Streamlit, Grad-CAM, benchmark multi-modeles, analyse seuils/erreurs et docs portfolio |
| [MultiCancer](projects/multicancer/README.md) | V1 complete + portfolio pack | Hub final des quatre projets avec cinq parcours, lazy loading, Grad-CAM, seuils et modes LungColon explicites |

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
|   |-- MULTICANCER_SCOPE.md
|   |-- MULTICANCER_ARCHITECTURE.md
|   |-- MULTICANCER_PROJECT_MATRIX.md
|   |-- MULTICANCER_DECISIONS.md
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
    |-- breast/
    |   |-- README.md
    |   |-- app.py
    |   |-- rccia_breast/
    |   |-- scripts/
    |   |-- tests/
    |   |-- docs/
    |   |-- data/
    |   `-- outputs/
    |-- metastasis/
    |   |-- README.md
    |   |-- app.py
    |   |-- rccia_metastasis/
    |   |-- scripts/
    |   |-- tests/
    |   |-- docs/
    |   |-- data/
    |   `-- outputs/
    `-- multicancer/
        |-- README.md
        |-- app.py
        |-- multicancer/
        |-- tests/
        `-- docs/
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

V2.1 : analyse des erreurs sur EfficientNet-B0, **130 erreurs sur 1 481 images test**, avec **88 false positives** (`benign -> malignant`) et **42 false negatives** (`malignant -> benign`). L'analyse inclut les erreurs par grossissement, les patients concentrant le plus d'erreurs et des exemples Grad-CAM locaux non versionnes.

Pack portfolio Breast :

- [Project summary](projects/breast/docs/PROJECT_SUMMARY.md)
- [Interview pitch](projects/breast/docs/INTERVIEW_PITCH.md)
- [LinkedIn draft](projects/breast/docs/LINKEDIN_DRAFT.md)
- [Release notes V2.1](projects/breast/docs/RELEASE_NOTES_V2_1.md)

## Lancer Le Projet Metastasis

```powershell
cd projects\metastasis
..\..\.venv\Scripts\streamlit.exe run app.py
```

V1 reelle : dataset Kaggle `tyson04/pcam-validate`, subset PCam HDF5 equilibre de 5 000 patches converti en ImageFolder, baseline ResNet18 pre-entrainee, accuracy test **0.9040**, ROC-AUC **0.9598**, PR-AUC **0.9616**. Ces resultats restent experimentaux sur subset public, sans validation clinique.

V2 : benchmark ResNet18 / MobileNetV3 small / EfficientNet-B0 sur le meme subset PCam 96x96. Meilleur score observe : EfficientNet-B0 avec **0.9320** accuracy test, **0.9320** macro F1, **0.9762** ROC-AUC et **0.9780** PR-AUC. ResNet18 obtient le meilleur recall `metastatic` (**0.9307**) et MobileNetV3 small est le plus rapide a entrainer. Voir [projects/metastasis/docs/V2_MODEL_COMPARISON.md](projects/metastasis/docs/V2_MODEL_COMPARISON.md).

V2.1 : analyse d'erreurs et de seuils sur EfficientNet-B0. Au seuil 0.50 : **699 correctes / 750**, **24 false positives**, **27 false negatives**, ROC-AUC **0.9762**, PR-AUC **0.9780**. L'analyse montre le compromis seuil bas / meilleur recall `metastatic` versus seuil haut / moins de faux positifs. Voir [projects/metastasis/docs/V2_ERROR_ANALYSIS.md](projects/metastasis/docs/V2_ERROR_ANALYSIS.md).

Pack portfolio Metastasis :

- [Project summary](projects/metastasis/docs/PROJECT_SUMMARY.md)
- [Interview pitch](projects/metastasis/docs/INTERVIEW_PITCH.md)
- [LinkedIn draft](projects/metastasis/docs/LINKEDIN_DRAFT.md)
- [Release notes V2.1](projects/metastasis/docs/RELEASE_NOTES_V2_1.md)

## Lancer Le Hub MultiCancer V1

Demo publique validee : https://rccia-multicancer.streamlit.app/

```powershell
.\.venv\Scripts\streamlit.exe run projects\multicancer\app.py
```

MultiCancer est le hub final du monorepo. Sa V1 integre Leukemia, Breast, Metastasis et
LungColon avec cinq parcours reels, chargement paresseux, predictions normalisees et
Grad-CAM. LungColon impose un choix explicite entre son checkpoint cinq classes et son
checkpoint binaire ; les deux modeles ne sont jamais charges simultanement.

Le hub route explicitement vers des pipelines specialises et ne construit pas de modele
medical universel.

Statut : **MultiCancer V1 complete / portfolio-ready**. Tag de reference :
`multicancer-v1`.

Pack portfolio MultiCancer :

- [Project summary](projects/multicancer/docs/PROJECT_SUMMARY.md)
- [Interview pitch](projects/multicancer/docs/INTERVIEW_PITCH.md)
- [LinkedIn drafts](projects/multicancer/docs/LINKEDIN_DRAFT.md)
- [Release notes V1](projects/multicancer/docs/RELEASE_NOTES_V1.md)

## Docs Globales

- [Roadmap](docs/ROADMAP.md)
- [Portfolio overview](docs/PORTFOLIO_OVERVIEW.md)
- [MultiCancer scope](docs/MULTICANCER_SCOPE.md)
- [MultiCancer architecture](docs/MULTICANCER_ARCHITECTURE.md)
- [MultiCancer project matrix](docs/MULTICANCER_PROJECT_MATRIX.md)
- [MultiCancer technical decisions](docs/MULTICANCER_DECISIONS.md)
- [MultiCancer V1 roadmap](projects/multicancer/docs/ROADMAP_V1.md)
- [MultiCancer V1.1 Leukemia adapter](projects/multicancer/docs/V1_LEUKEMIA_ADAPTER.md)
- [MultiCancer V1.2 Breast adapter](projects/multicancer/docs/V1_BREAST_ADAPTER.md)
- [MultiCancer V1.3 Metastasis adapter](projects/multicancer/docs/V1_METASTASIS_ADAPTER.md)
- [MultiCancer V1.4 LungColon adapter](projects/multicancer/docs/V1_LUNG_COLON_ADAPTER.md)
- [MultiCancer project summary](projects/multicancer/docs/PROJECT_SUMMARY.md)
- [MultiCancer interview pitch](projects/multicancer/docs/INTERVIEW_PITCH.md)
- [MultiCancer LinkedIn drafts](projects/multicancer/docs/LINKEDIN_DRAFT.md)
- [MultiCancer release notes V1](projects/multicancer/docs/RELEASE_NOTES_V1.md)
- [Leukemia project summary](projects/leukemia/docs/PROJECT_SUMMARY.md)
- [Leukemia interview pitch](projects/leukemia/docs/INTERVIEW_PITCH.md)
- [Leukemia LinkedIn draft](projects/leukemia/docs/LINKEDIN_DRAFT.md)
- [Breast V2 model comparison](projects/breast/docs/V2_MODEL_COMPARISON.md)
- [Breast V2 error analysis](projects/breast/docs/V2_ERROR_ANALYSIS.md)
- [Metastasis dataset guide](projects/metastasis/docs/DATASET_GUIDE.md)
- [Metastasis metrics guide](projects/metastasis/docs/METRICS_GUIDE.md)
- [Metastasis project summary](projects/metastasis/docs/PROJECT_SUMMARY.md)
- [Metastasis interview pitch](projects/metastasis/docs/INTERVIEW_PITCH.md)
- [Metastasis LinkedIn draft](projects/metastasis/docs/LINKEDIN_DRAFT.md)
- [Metastasis V2 model comparison](projects/metastasis/docs/V2_MODEL_COMPARISON.md)
- [Metastasis V2 error analysis](projects/metastasis/docs/V2_ERROR_ANALYSIS.md)
