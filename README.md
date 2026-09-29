# R.C.C.I.A - Computer Vision et ingenierie ML

**Comparer des classifieurs d'images biomedicales sans masquer leurs erreurs
derriere une accuracy globale.** Ce portfolio transforme quatre experiences
specialisees en pipelines de preparation, entrainement, evaluation et prediction,
reunis dans le hub Streamlit MultiCancer.

Il permet aux recruteurs, developpeurs et apprenants d'examiner une demarche ML de
bout en bout. **Ce n'est ni un dispositif medical, ni un outil de diagnostic,
ni une IA cliniquement validee.**

![MultiCancer : prediction et exploration des seuils Metastasis](projects/multicancer/docs/assets/metastasis-threshold-analysis.png)

[Captures des cinq parcours](projects/multicancer/README.md#apercu-streamlit) |
[Installation et actifs requis](docs/LOCAL_SETUP.md) |
[Validation locale datee](docs/PORTFOLIO_VALIDATION.md) |
[Architecture](docs/MULTICANCER_ARCHITECTURE.md)

## Ce qui est livre

- Quatre pipelines : **Leukemia, LungColon, Breast et Metastasis** ; cinq parcours
  dans le hub, car LungColon possede un mode cinq classes et un mode binaire.
- Preparation ImageFolder/HDF5, prediction CLI, rapports CSV/JSON, matrices de
  confusion et comparaison **ResNet18 / MobileNetV3 Small / EfficientNet-B0**.
- Probabilites, Grad-CAM et analyse d'erreurs ; analyse Breast par patient et
  grossissement ; exploration des seuils Metastasis.
- Adaptateurs specialises, resultat commun et chargement paresseux : au plus un
  modele charge **par session**, decharge avant changement de projet ou de mode.
- Tests automatises des pipelines, contrats, erreurs, uploads et parcours Streamlit.

**Stack :** Python, PyTorch/torchvision, NumPy, pandas, scikit-learn, Pillow/OpenCV,
h5py, Matplotlib/Seaborn, Streamlit et pytest. Les architectures sont preentrainees
puis adaptees aux classes du projet ; aucun modele universel de cancer n'est revendique.

## Resultats experimentaux

Resultats historiques documentes, pas de nouveau benchmark lors de la finition
portfolio. Les protocoles et jeux de test different : **ne pas classer les projets
entre eux a partir de ces scores**.

| Experience | Test | Resultat de reference | Details |
| --- | ---: | --- | --- |
| Leukemia V1, ResNet18 | 1 601 images | Accuracy 91,69 % | [Leukemia](projects/leukemia/README.md) |
| LungColon cinq classes, EfficientNet-B0 | 3 750 images | Accuracy 99,92 %, 3 erreurs | [Benchmark](projects/lung_colon/docs/V2_MODEL_COMPARISON.md) |
| Breast, EfficientNet-B0 | 1 481 images | Accuracy 91,22 %, macro F1 90,04 %, recall malignant 95,68 % | [Benchmark patient-aware](projects/breast/docs/V2_MODEL_COMPARISON.md) |
| Metastasis, EfficientNet-B0 | 750 patches | Accuracy 93,20 %, ROC-AUC 0,9762, average precision 0,9780 | [Benchmark PCam subset](projects/metastasis/docs/V2_MODEL_COMPARISON.md) |

### Ce que l'analyse apporte

**Breast :** separation par patient (55 / 11 / 15 patients en train / validation /
test, sans intersection documentee). L'analyse releve **130 erreurs, dont 76 sur
un seul patient** : une moyenne globale masque une forte variabilite.
[Analyse detaillee](projects/breast/docs/V2_ERROR_ANALYSIS.md).

**Metastasis :** abaisser le seuil de **0,50 a 0,30** fait passer les faux negatifs
de **27 a 13**, mais les faux positifs de **24 a 69**. Le hub illustre ce compromis
sans proposer de seuil medical. Le champ historique `pr_auc` correspond a
`average_precision_score`, pas a une integration trapezoidale.
[Analyse des seuils](projects/metastasis/docs/V2_ERROR_ANALYSIS.md).

## Limites

- Aucune validation clinique ni validation sur une cohorte externe independante.
- LC25000 contient des images augmentees ; le split par image ne garantit pas
  l'independance des sources. Les scores peuvent donc etre optimistes.
- Le protocole Leukemia ne garantit pas un groupement par patient ; Metastasis
  utilise 5 000 patches issus de la validation PCam, redivises localement.
- Comparaisons courtes et exploratoires : choisir modele ou seuil en regardant
  le test peut biaiser les conclusions. Aucun gain clinique ou utilisateur mesure.
- Une forte probabilite et une carte Grad-CAM ne garantissent pas une prediction
  correcte. La capture Breast conserve volontairement une erreur a forte confiance.

## Essayer le projet

### 1. Consulter l'interface sans poids

Depuis un clone, utiliser **Python 3.11** (version locale validee) :

```powershell
git clone https://github.com/amiralfefe/R.C.C.I.A.git
cd R.C.C.I.A
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run projects/multicancer/app.py
```

Sous Linux/macOS, utiliser `python3.11 -m venv .venv` puis `.venv/bin/python`.
Le registre, les limites et les metriques sont consultables sans dataset ni token.
Sans checkpoint, l'inference est indisponible : ce n'est pas une prediction simulee.

### 2. Effectuer une inference reelle

Il faut le checkpoint compatible du parcours et une image publique adaptee.
**Les cinq poids ne sont pas dans Git et le depot Hugging Face est prive** :
un clone seul ne donne pas acces aux modeles. Aucun token personnel n'est fourni.
Les chemins attendus et les options de reproduction sont dans
[le guide local](docs/LOCAL_SETUP.md#inference-avec-checkpoints).

### Demo hebergee : disponibilite non verifiee

[URL historique MultiCancer](https://rccia-multicancer.streamlit.app/).
Le dernier controle applicatif documente, le **20 septembre 2026**, constatait
un refus d'acces HF aux checkpoints. La finition du **29 septembre 2026** ne
revalide pas ce service et ne modifie aucun secret distant. Les captures et les
rapports restent consultables sans cette demo.
[Suivi de deploiement](deploy/streamlit-multicancer/DEPLOYMENT_VALIDATION.md).

## Tests et reproductibilite

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q -ra
.\.venv\Scripts\python.exe -m pip check
```

Le [rapport de validation](docs/PORTFOLIO_VALIDATION.md) distingue l'etat local
avec actifs et une copie des sources sans donnees ni checkpoints. Certaines QA
reelles sont ignorees lorsque ces actifs manquent ; les smoke tests synthetiques
ne sont pas des resultats ML reels. Pas de CI distante revendiquee.

## Organisation

```text
projects/
  leukemia/       pipeline binaire et interpretabilite
  lung_colon/     classification cinq classes et binaire
  breast/         split patient-aware, grossissements et erreurs
  metastasis/     PCam, ROC/PR et seuils
  multicancer/    hub, schemas, adaptateurs et tests d'integration
  rccia_common/   validation des images et des chemins de donnees
deploy/          bootstrap et packaging, distincts des pipelines ML
docs/            architecture, guides, limites et preuves datees
```

Les sous-projets conservent leurs commandes de reproduction, benchmarks et
rapports historiques. Les datasets, outputs, checkpoints, secrets et `.venv`
restent hors Git. Le tag historique `multicancer-v1` reste sur `a51772d` ; il ne
contient pas les correctifs ulterieurs. Le depot ne definit pas encore de licence
de reutilisation : ne pas supposer une autorisation de redistribution.

## Pour approfondir

- [README MultiCancer et captures](projects/multicancer/README.md)
- [Contrats et cycle de vie des modeles](docs/MULTICANCER_ARCHITECTURE.md)
- [Matrice des projets](docs/MULTICANCER_PROJECT_MATRIX.md)
- [Confidentialite et usage des images publiques](docs/PRIVACY.md)
- [Audit technique du 20 septembre, historique](docs/AUDIT_2026-09-20.md)
- [Roadmap et jalons historiques](docs/ROADMAP.md)
