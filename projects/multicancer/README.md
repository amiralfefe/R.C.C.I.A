# R.C.C.I.A MultiCancer

MultiCancer est le hub de synthese du monorepo R.C.C.I.A. Il regroupe les projets
Leukemia, LungColon, Breast et Metastasis au travers d'une architecture d'adaptateurs
specialises.

> Demonstrateur educatif / portfolio uniquement. Aucune validation clinique, aucun
> diagnostic medical et aucune recommandation medicale.

## Statut

**V0 - Scope, architecture et initialisation legere.**

La V0 fournit la documentation, un registre de metadonnees, des schemas communs
conceptuels, une page Streamlit informative et des tests rapides. Elle ne charge aucun
checkpoint, ne lit aucun dataset et ne realise aucune prediction.

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

Les adaptateurs seront ajoutes en V1. Un seul modele sera charge a la fois et l'absence
de checkpoint sera geree sans bloquer le hub.

## Lancer La Page V0

Depuis la racine du monorepo :

```powershell
.\.venv\Scripts\streamlit.exe run projects\multicancer\app.py
```

La page fonctionne sans outputs, checkpoint ou dataset local.

## Documentation

- [Scope](../../docs/MULTICANCER_SCOPE.md)
- [Architecture](../../docs/MULTICANCER_ARCHITECTURE.md)
- [Project matrix](../../docs/MULTICANCER_PROJECT_MATRIX.md)
- [Technical decisions](../../docs/MULTICANCER_DECISIONS.md)
- [V1 roadmap](docs/ROADMAP_V1.md)

## Roadmap Courte

1. stabiliser le registre et les schemas ;
2. implementer un adaptateur par projet ;
3. integrer la selection et le chargement paresseux dans Streamlit ;
4. normaliser predictions, probabilites et explications ;
5. tester les erreurs et checkpoints absents.

MultiCancer n'est pas une IA de diagnostic multi-cancer. C'est une plateforme portfolio
qui expose plusieurs pipelines experimentaux, leurs performances et leurs limites.
