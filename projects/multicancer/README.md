# R.C.C.I.A MultiCancer

MultiCancer est le hub de synthese du monorepo R.C.C.I.A. Il regroupe les projets
Leukemia, LungColon, Breast et Metastasis au travers d'une architecture d'adaptateurs
specialises.

> Demonstrateur educatif / portfolio uniquement. Aucune validation clinique, aucun
> diagnostic medical et aucune recommandation medicale.

## Statut

**V1.3 - Leukemia + Breast + Metastasis Adapters.**

La V0 a defini le scope, le registre et l'architecture. La V1.1 ajoute le contrat commun,
un gestionnaire garantissant un seul modele en memoire et le premier adaptateur reel :
Leukemia. La V1.2 ajoute Breast avec le meme contrat et un contexte patient-aware visible.
La V1.3 ajoute Metastasis et une couche de decision par seuil separee de l'inference.

## V1.1 - Leukemia Adapter

Fonctionnalites :

- selection explicite du projet ;
- detection du checkpoint Leukemia local sans le versionner ;
- chargement paresseux et idempotent ;
- prediction normalisee `normal` / `leukemia_blast` ;
- probabilites par classe ;
- Grad-CAM optionnel reutilisant le pipeline existant ;
- erreurs controlees pour checkpoint absent/incompatible et image invalide ;
- dechargement du modele lors d'un changement de projet ;
- limites et disclaimers global/Leukemia visibles.

Breast, Metastasis et LungColon restent visibles mais ne chargent aucun modele dans
cette phase.

## V1.2 - Breast Adapter

Fonctionnalites :

- checkpoint EfficientNet-B0 local charge a la demande ;
- prediction `benign` / `malignant` normalisee ;
- probabilites et Grad-CAM en memoire ;
- split patient-aware, 81 patients et overlap patient nul visibles ;
- resultats par grossissement 40X / 100X / 200X / 400X ;
- analyse des erreurs et concentration par patient contextualisees ;
- bascule Leukemia / Breast avec unload de l'ancien modele.

Metastasis et LungColon restent visibles mais ne chargent aucun modele dans cette phase.

## V1.3 - Metastasis Adapter

Fonctionnalites :

- checkpoint EfficientNet-B0 local et preprocessing `96x96` ;
- prediction `non_metastatic` / `metastatic` par argmax ;
- probabilites brutes et Grad-CAM ;
- ROC-AUC, PR-AUC et resultats FP/FN documentes ;
- `ThresholdDecision` distinct du `PredictionResult` ;
- slider exploratoire `0.30` a `0.70` sans nouvelle inference ;
- argmax original toujours affiche ;
- aucun seuil medical recommande.

LungColon reste visible mais ne charge aucun modele dans cette phase.

## Projets Regroupes

- Leukemia : cellules sanguines, `normal` vs `leukemia_blast` ;
- LungColon : histopathologie poumon/colon, cinq classes et mode binaire ;
- Breast : BreakHis, `benign` vs `malignant`, split patient-aware ;
- Metastasis : patches PCam, `non_metastatic` vs `metastatic`, ROC et seuils.

## Pourquoi Pas Un Modele Universel ?

Ces projets utilisent des modalites, classes, resolutions, datasets et protocoles
differents. Les melanger dans un seul modele en V0 masquerait ces differences et
encouragerait des comparaisons trompeuses. MultiCancer conserve donc un pipeline
specialise par tache et normalise seulement leur presentation.

## Architecture Prevue

```text
Hub Streamlit
  -> registre du projet choisi
  -> adaptateur specialise
  -> preprocessing et checkpoint propres au projet
  -> resultat commun + Grad-CAM optionnel + limites
```

Leukemia, Breast et Metastasis sont integres avec un socle Torchvision commun. Un seul modele est
charge a la fois et l'absence d'un checkpoint ne bloque pas le hub.

## Lancer Le Hub V1.3

Depuis la racine du monorepo :

```powershell
.\.venv\Scripts\streamlit.exe run projects\multicancer\app.py
```

La page reste consultable sans outputs, checkpoint ou dataset local. Les predictions
necessitent les checkpoints locaux des projets selectionnes.

## Documentation

- [Scope](../../docs/MULTICANCER_SCOPE.md)
- [Architecture](../../docs/MULTICANCER_ARCHITECTURE.md)
- [Project matrix](../../docs/MULTICANCER_PROJECT_MATRIX.md)
- [Technical decisions](../../docs/MULTICANCER_DECISIONS.md)
- [V1 roadmap](docs/ROADMAP_V1.md)
- [Leukemia adapter V1.1](docs/V1_LEUKEMIA_ADAPTER.md)
- [Breast adapter V1.2](docs/V1_BREAST_ADAPTER.md)
- [Metastasis adapter V1.3](docs/V1_METASTASIS_ADAPTER.md)

## Roadmap Courte

1. Leukemia : integre en V1.1 ;
2. Breast : integre en V1.2 ;
3. Metastasis : integre en V1.3 avec ROC, probabilite positive et seuils ;
4. LungColon : prochain adaptateur, avec selection des modes 5 classes et binaire.

MultiCancer n'est pas une IA de diagnostic multi-cancer. C'est une plateforme portfolio
qui expose plusieurs pipelines experimentaux, leurs performances et leurs limites.
