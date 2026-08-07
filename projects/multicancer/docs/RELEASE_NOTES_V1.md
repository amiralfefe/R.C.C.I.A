# MultiCancer - Release Notes V1

## Resume

MultiCancer V1 finalise le hub transversal de R.C.C.I.A. La release regroupe quatre
pipelines specialises et cinq parcours explicites sans fusionner les datasets, les
classes ou les modeles.

Tag de reference : `multicancer-v1`.

## Historique

### V0 - Scope Et Architecture

- Definition du hub specialise.
- Registre de metadonnees sans checkpoint.
- Schemas communs.
- Decisions d'architecture et matrice de projets.
- Streamlit consultable sans modele local.

### V1.1 - Leukemia

- Premier adaptateur reel.
- Lazy loading ResNet18.
- Prediction normalisee et Grad-CAM.
- Gestion checkpoint absent et image invalide.

### V1.2 - Breast

- Adaptateur EfficientNet-B0.
- Contexte BreakHis patient-aware.
- Grossissements et limites par patient.
- Bascule Leukemia / Breast avec unload.

### V1.3 - Metastasis

- Adaptateur EfficientNet-B0 en `96x96`.
- ROC-AUC / PR-AUC et tableau FP/FN.
- `ThresholdDecision` distinct de `PredictionResult`.
- Slider exploratoire sans nouvelle inference.

### V1.4 - LungColon

- Mode cinq classes EfficientNet-B0.
- Mode binaire ResNet18.
- Checkpoints et classes strictement separes.
- Unload et suppression des sorties obsoletes au changement de mode.

### V1.5 - Final QA Et Portfolio Pack

- QA reelle des cinq parcours avec checkpoints locaux.
- QA Streamlit AppTest complete sans exception.
- Verification probabilites, Grad-CAM, seuil et lifecycle.
- Captures portfolio reelles.
- Resume projet, pitch entretien et brouillons LinkedIn.
- README et roadmaps finalises.

## Fonctionnalites Finales

- quatre projets specialises ;
- cinq parcours reels avec LungColon multiclass et binary ;
- selection explicite du projet et du mode ;
- lazy loading et unload idempotent ;
- un seul modele charge a la fois ;
- probabilites normalisees ;
- Grad-CAM en memoire ;
- exploration de seuil Metastasis sans mutation de l'inference ;
- erreurs communes controlees ;
- application Streamlit ;
- 120 tests automatises passes localement, dont six QA AppTest reels et CI-safe.

## QA Finale

La QA locale a confirme :

| Parcours | Checkpoint | Probabilites | Grad-CAM | Lifecycle |
| --- | --- | ---: | --- | --- |
| Leukemia | ResNet18 | 2, somme 1 | 224x224x3 | valide |
| Breast | EfficientNet-B0 | 2, somme 1 | 224x224x3 | valide |
| Metastasis | EfficientNet-B0 | 2, somme 1 | 96x96x3 | valide |
| LungColon multiclass | EfficientNet-B0 | 5, somme 1 | 224x224x3 | valide |
| LungColon binary | ResNet18 | 2, somme 1 | 224x224x3 | valide |

Le scenario AppTest a parcouru Leukemia, Breast, Metastasis, LungColon multiclass,
LungColon binary puis un retour vers Leukemia. Il a confirme l'absence de sortie
obsolette et le dechargement du modele precedent.

## Limites

- Les datasets publics et leurs protocoles sont heterogenes.
- Les metriques ne sont pas directement comparables entre projets.
- Aucun dataset n'apporte une validation clinique externe ou multi-centres.
- Une forte confiance peut accompagner une prediction incorrecte.
- Grad-CAM est exploratoire et non causal.
- Le seuil Metastasis sert uniquement a illustrer un compromis statistique.
- Les scores LC25000 locaux ne prouvent pas une generalisation.

## Fichiers Volontairement Exclus

Les elements suivants restent locaux et non versionnes :

- `data/` ;
- `outputs/` ;
- checkpoints `.pt`, `.pth`, `.ckpt` ;
- tokens, `.env`, `kaggle.json` et `access_token` ;
- `.venv/` ;
- images source et sorties Grad-CAM temporaires.

## Cadre D'Utilisation

MultiCancer est un demonstrateur educatif et portfolio. Il ne fournit aucun diagnostic,
aucune recommandation medicale et aucune validation clinique. Le hub ne choisit pas
automatiquement un cancer ou une modalite.
