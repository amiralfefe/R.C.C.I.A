# Validation de finition portfolio - 29 septembre 2026

## Etat et perimetre

Base : branche `main`, parent audite `a323780ae5a7fbf52c12015a6c84d4aab921c1c2`.
Au debut : 24 fichiers suivis modifies et 6 nouveaux fichiers, identiques a
l'Audit 2. Cette validation porte sur ces correctifs consolides et la finition
documentaire, pas seulement sur le commit parent ni sur le deploiement public.

Les protections contre suppression accidentelle, decodeurs non autorises,
resultats obsoletes, recomputation Grad-CAM et collisions du downloader sont
conservees. Aucun nouveau modele, poids, benchmark ou entrainement reel.
La suite execute ses smoke tests sur de petits jeux synthetiques, qui ne
constituent pas des performances experimentales.

## Resultats de cette execution

| Controle | Resultat du 29 septembre |
| --- | --- |
| Suite monorepo avec actifs locaux | **201 passed, 0 failed, 0 skipped**, 86,95 s |
| Meme suite sur copie des sources seules | **193 passed, 0 failed, 8 skipped**, 73,58 s |
| AppTest sans poids : cinq parcours | **5/5**, aucune exception, checkpoint manquant signale, aucun modele charge |
| AppTest avec poids | Inclus dans les 201 tests : predictions et Grad-CAM des cinq parcours, transitions et regressions |
| `pip check` | No broken requirements found |
| Imports principaux | Quatre modules model, ModelManager et decodeur commun importes sans erreur |
| Syntaxe Python | 122 fichiers parses avec `ast.parse`, sans erreur |
| Configuration | pyproject et configuration Streamlit parses avec `tomllib` |
| `git diff --check` | Reussi ; avertissements LF/CRLF sans erreur de diff |
| Lint | Ruff non installe ; aucune execution de lint/type checking revendiquee |
| Documentation | 72 Markdown controles, aucun lien relatif rendu casse |
| Publication | 244 fichiers suivis ou candidats controles par noms/motifs : aucun secret ni poids detecte ; controles d'ignore reussis |

La copie source-only a ete constituee dans un repertoire temporaire neuf a partir
des fichiers suivis et des nouveaux fichiers candidats, sans Git, dataset,
checkpoint, output reel ou venv. Elle utilise **le venv existant** : ce n'est
ni une installation vierge des dependances ni un clone telecharge depuis GitHub.
Des documents ont ensuite ete completes ; aucun code Python n'a change apres
les deux passes.

Les 8 skips sont les cinq cas parametres de prediction reelle et les trois QA
Metastasis (slider/Grad-CAM, changement d'upload, echec d'inference). Motif :
`Local QA assets unavailable` ; aucun echec n'est transforme en skip.
Un premier essai de commande AppTest inline a echoue sur le quoting PowerShell,
avant lancement de l'application ; la commande corrigee a valide les cinq parcours.

Les 120 tests du tag V1 et les 201 tests du 20 septembre restent historiques.
Le total actuel reste 201 : cette finition documentaire conserve la collecte
elargie de l'audit precedent sans ajouter de test. L'ecart 201/193 correspond
uniquement aux 8 QA dependantes des actifs.

## Commandes executees

Depuis la racine avec les actifs locaux :

```powershell
.\.venv\Scripts\python.exe -m pytest -q -ra
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

Depuis la copie temporaire : meme interpreteur local, avec
`-B -m pytest -q -ra -p no:cacheprovider`.
Le smoke AppTest utilise `AppTest.from_file('projects/multicancer/app.py')`,
selectionne les quatre projets et les deux modes LungColon, puis verifie
absence d'exception, statut `missing` et `is_loaded == False`.
Il remplace ici un lancement de serveur persistant ; pas de nouveau test visuel
sur navigateur ou telephone physique pendant cette finition.

## Environnement observe

Windows, Python 3.11.8 ; versions lues dans le venv, sans mise a jour de paquet
pendant cette mission :

| Paquet | Version |
| --- | --- |
| torch / torchvision | 2.12.1 / 0.27.1 |
| Streamlit | 1.58.0 |
| Pillow | 12.3.0 |
| pytest | 9.1.1 |
| NumPy / pandas | 2.4.6 / 3.0.3 |
| scikit-learn | 1.9.0 |
| h5py | 3.16.0 |
| huggingface-hub | 1.27.0 |

Ce tableau n'est pas un lockfile. Les requirements de deploiement sont distincts.
Une installation complete et un build Docker sur machine vierge restent non
verifies. Les avis de dependances restants du [precedent audit](AUDIT_2026-09-20.md)
ne sont pas declares resolus par cette finition ; pas de certification de securite.

## Presentation et reproduction

README racine : probleme, livrables, erreurs, limites et capture existante.
Architecture : composants reels, sans cibles V0 annoncees comme encore absentes.
Guide local : Python 3.11, chemins relatifs, consultation sans poids, actifs requis.
README hub et rapport public : historiques dates, aucune disponibilite garantie.

Un clone permet de lire rapports/captures et de parcourir le hub apres installation.
Une inference reelle exige les checkpoints compatibles et une image publique
appropriee ; les poids HF restent prives. Aucun dataset complet requis pour la
consultation. Voir [LOCAL_SETUP.md](LOCAL_SETUP.md).

## Fichiers consolides

Les 30 fichiers de depart sont listes dans
[l'audit technique](AUDIT_2026-09-20.md#liste-exacte-des-fichiers-modifies-ou-ajoutes).
Tous sont conserves ; le README racine est reorganise. Autres changements :

- `docs/MULTICANCER_ARCHITECTURE.md` : architecture livree ;
- `docs/MULTICANCER_SCOPE.md` : etiquette historique et liens actuels ;
- `projects/multicancer/README.md` : tests dates, poids et disponibilite ;
- `projects/leukemia/docs/SETUP_WINDOWS.md` : racine du clone, pas chemin personnel ;
- `deploy/streamlit-multicancer/DEPLOYMENT_VALIDATION.md` : avertissement historique ;
- `docs/LOCAL_SETUP.md` : guide de consultation et d'inference ;
- `docs/PORTFOLIO_VALIDATION.md` : cette preuve locale datee.

## Publication et limites

**Aucun push, changement de tag, modification de branche distante ou de service
Streamlit/HF n'est effectue par cette mission.** La branche de deploiement n'est
pas automatiquement alignee par une modification de main. Le tag ML reste fige.
Un commit local, s'il est cree, n'est pas une publication GitHub.
Le controle de secrets est limite aux motifs usuels et noms de fichiers ; il ne
certifie pas tout l'historique Git. Le perimetre de finition contient 37 fichiers
(29 suivis modifies et 8 nouveaux), sans dataset ni checkpoint.

Demo publique : **non verifiee aujourd'hui** ; dernier constat applicatif
documente : acces HF refuse le 20 septembre. La resolution des secrets distants
appartient au proprietaire et ne conditionne pas la consultation des preuves.
Datasets, checkpoints, outputs, tokens, caches et venv restent hors Git.
Licence de reutilisation non choisie, CI distante absente, aucune validation
clinique/externe ni certification de charge, RGPD ou accessibilite complete.

Verdict local : **A - Portfolio-ready comme demonstrateur educatif documente**,
pas comme service de production ou outil clinique. Publication soumise a
autorisation ; disponibilite publique non promise.
