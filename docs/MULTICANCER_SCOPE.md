# MultiCancer Scope

Document de cadrage historique V0. Pour les composants livres et leur statut,
voir [l'architecture actuelle](MULTICANCER_ARCHITECTURE.md) et la
[validation locale datee](PORTFOLIO_VALIDATION.md).

## Vision

MultiCancer est le hub final du monorepo R.C.C.I.A. Il rassemble quatre pipelines
specialises sans pretendre qu'un seul modele peut traiter correctement toutes les
modalites, tous les tissus et toutes les taches.

Le hub doit rendre le portfolio plus facile a explorer, tout en conservant les choix de
preprocessing, les classes, les checkpoints, les metriques et les limites propres a
chaque projet.

> MultiCancer est un demonstrateur educatif et portfolio uniquement. Il ne fournit
> aucun diagnostic, aucune recommandation medicale et ne dispose d'aucune validation
> clinique.

## Probleme Resolu

Les projets Leukemia, LungColon, Breast et Metastasis sont autonomes et utilisent des
datasets et protocoles differents. MultiCancer fournit une entree commune pour les
presenter et, en V1, appeler explicitement le pipeline specialise choisi par
l'utilisateur.

Cette centralisation ne rend pas les projets interchangeables. Elle expose au contraire
leurs differences afin d'eviter une lecture naive des resultats.

## Ce Que MultiCancer Fera

- selection explicite d'un sous-projet ;
- routage vers le pipeline specialise correspondant ;
- prediction au travers d'un adaptateur propre au projet ;
- affichage normalise de la classe, de la confiance et des probabilites ;
- Grad-CAM lorsque le modele et l'adaptateur le permettent ;
- affichage des metriques documentees pour le projet ;
- affichage des limites et avertissements propres au projet ;
- comparaison methodologique des datasets, splits, taches et protocoles.

## Ce Que MultiCancer Ne Fera Pas

- diagnostic medical ;
- modele universel couvrant tous les cancers ;
- comparaison naive des accuracies ;
- fusion automatique des datasets ;
- recommandation medicale ;
- validation clinique.

## Utilisateurs Cibles

- recruteurs et equipes techniques ;
- reviewers GitHub ;
- visiteurs du portfolio ;
- demonstrations techniques ;
- apprentissage et experimentation personnels.

## Perimetre V0

La V0 couvre uniquement le cadrage, l'architecture, un registre de metadonnees, des
schemas communs conceptuels, une page Streamlit informative et des tests rapides. Elle
ne charge aucun modele et n'accede a aucune donnee.

## Perimetre V1

La V1 ajoutera un hub Streamlit interactif et un adaptateur par projet. Un seul modele
sera charge a la fois, apres selection explicite de la tache par l'utilisateur.

## Hors Scope Initial

- entrainement commun ou multi-datasets ;
- API clinique ;
- authentification ;
- stockage de donnees patient ;
- traitement de donnees personnelles ;
- deploiement medical ou reglementaire ;
- routage automatique opaque de la modalite.

## Comparaison Des Projets

La matrice factuelle est disponible dans
[MULTICANCER_PROJECT_MATRIX.md](MULTICANCER_PROJECT_MATRIX.md).

Les metriques ne doivent pas etre classees du meilleur au moins bon : datasets, taches,
splits, resolutions, tailles d'echantillon et niveaux de difficulte different. Une
accuracy plus elevee dans un projet ne prouve pas qu'un modele est meilleur dans un
autre contexte.
