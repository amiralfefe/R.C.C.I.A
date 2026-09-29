# Installation locale et actifs requis

Les commandes partent de la racine du clone, sans chemin personnel obligatoire.
Python **3.11** est la cible validee ; `>=3.10` dans pyproject n'est pas une
matrice de tests de toutes les versions.

## Consultation sans poids

```powershell
git clone https://github.com/amiralfefe/R.C.C.I.A.git
cd R.C.C.I.A
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run projects/multicancer/app.py
```

Sous Linux/macOS : `python3.11 -m venv .venv`, puis remplacer
`.\.venv\Scripts\python.exe` par `.venv/bin/python`. La validation de cette
finition est effectuee sous Windows ; les commandes POSIX ne sont pas testees.
L'activation du venv n'est pas necessaire avec ces chemins explicites.

Le hub affiche les projets, metriques historiques et limites. Les checkpoints
absents sont signales ; l'analyse ne fonctionne pas tant qu'ils manquent.
Ne pas utiliser `deploy/streamlit-multicancer/app.py` pour ce mode : ce bootstrap
exige de preparer les cinq poids au demarrage. Aucun dataset ni token ne sont
necessaires pour consulter directement `projects/multicancer/app.py`.

Les dependances sont volumineuses (PyTorch notamment). Le fichier racine contient
des bornes, pas un lockfile. Voir les versions testees et les limites du controle
d'installation dans le [rapport date](PORTFOLIO_VALIDATION.md).

## Inference avec checkpoints

Une image PNG/JPEG publique adaptee et le poids du parcours suffisent ; aucun
dataset complet n'est necessaire. Ne charger que des checkpoints de confiance.

| Parcours | Chemin depuis la racine |
| --- | --- |
| Leukemia | `projects/leukemia/outputs/best_model.pt` |
| Breast | `projects/breast/outputs/model_comparison/efficientnet_b0/best_model.pt` |
| Metastasis | `projects/metastasis/outputs/model_comparison/efficientnet_b0/best_model.pt` |
| LungColon cinq classes | `projects/lung_colon/outputs/model_comparison/efficientnet_b0/best_model.pt` |
| LungColon binaire | `projects/lung_colon/outputs/binary_resnet18/best_model.pt` |

Ces fichiers ne sont pas dans Git. Le depot `amiralfefe/rccia-multicancer-models`
est prive : un recruteur sans autorisation ne peut pas les telecharger. Obtenir
des poids compatibles aupres du proprietaire, ou suivre le protocole du sous-projet
pour produire ses propres poids. Cela demande donnees et calcul et ne garantit
pas les chiffres historiques a l'identique.

Le [bootstrap HF](../deploy/hf-multicancer/README.md) est reserve aux utilisateurs
deja autorises ; il requiert ses dependances de deploiement et `HF_MODEL_REPO_ID`
/ `HF_TOKEN` dans l'environnement serveur. Ne jamais communiquer ou versionner
un token. Ce bootstrap est inutile si les poids sont deja presents localement.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q -ra
.\.venv\Scripts\python.exe -m pip check
```

Les tests unitaires et smoke tests fabriquent de petits exemples synthetiques :
ce ne sont pas des benchmarks reels. Les QA de
`projects/multicancer/tests/test_portfolio_qa.py` necessitent les checkpoints et
les exemples nommes dans `LOCAL_CASES`. Elles sont ignorees avec le motif
`Local QA assets unavailable` lorsqu'ils manquent. `-ra` affiche les raisons.
Pas de CI distante configuree ; ne pas assimiler ces resultats a un badge CI.

## Reproduire les experiences (optionnel)

- [Leukemia](../projects/leukemia/README.md)
- [LungColon](../projects/lung_colon/README.md)
- [Breast et split patient-aware](../projects/breast/README.md)
- [Metastasis et HDF5](../projects/metastasis/README.md)

Les anciens exemples `C:\VSCODE\datasets\...` sont des chemins de la machine
d'origine : remplacer leur valeur par son propre dossier, par exemple
`../datasets/breakhis` depuis la racine. Apres un `cd` dans un sous-projet,
adapter ce chemin ou utiliser un chemin absolu choisi par l'utilisateur.
Sources et sorties doivent etre disjointes ; `--overwrite` remplace les sorties.

Aucune base de donnees, authentification ou execution d'entrainement n'est
requise pour parcourir le hub. Donnees, poids, outputs, caches, secrets et
environnement virtuel restent locaux et ignores par Git.
