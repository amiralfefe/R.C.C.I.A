# R.C.C.I.A MultiCancer

MultiCancer est le hub de synthese du monorepo R.C.C.I.A. Il regroupe les projets
Leukemia, LungColon, Breast et Metastasis au travers d'une architecture d'adaptateurs
specialises.

> Demonstrateur educatif / portfolio uniquement. Aucune validation clinique, aucun
> diagnostic medical et aucune recommandation medicale.

## Statut

**V1.1 - Common Adapter Contract + Leukemia End-to-End Integration.**

La V0 a defini le scope, le registre et l'architecture. La V1.1 ajoute le contrat commun,
un gestionnaire garantissant un seul modele en memoire et le premier adaptateur reel :
Leukemia.

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

Leukemia est integre. Les adaptateurs suivants conserveront le meme contrat et un seul
modele sera charge a la fois. L'absence de checkpoint ne bloque pas le hub.

## Lancer Le Hub V1.1

Depuis la racine du monorepo :

```powershell
.\.venv\Scripts\streamlit.exe run projects\multicancer\app.py
```

La page reste consultable sans outputs, checkpoint ou dataset local. Une prediction
Leukemia necessite le checkpoint local `projects/leukemia/outputs/best_model.pt`.

## Documentation

- [Scope](../../docs/MULTICANCER_SCOPE.md)
- [Architecture](../../docs/MULTICANCER_ARCHITECTURE.md)
- [Project matrix](../../docs/MULTICANCER_PROJECT_MATRIX.md)
- [Technical decisions](../../docs/MULTICANCER_DECISIONS.md)
- [V1 roadmap](docs/ROADMAP_V1.md)
- [Leukemia adapter V1.1](docs/V1_LEUKEMIA_ADAPTER.md)

## Roadmap Courte

1. Leukemia : integre en V1.1 ;
2. Breast : prochain adaptateur ;
3. Metastasis : adapter ROC, probabilite positive et seuils ;
4. LungColon : integrer en dernier les modes 5 classes et binaire.

MultiCancer n'est pas une IA de diagnostic multi-cancer. C'est une plateforme portfolio
qui expose plusieurs pipelines experimentaux, leurs performances et leurs limites.
