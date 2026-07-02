# Portfolio Overview - R.C.C.I.A

## Resume Recruteur

R.C.C.I.A est un monorepo portfolio de projets IA/data en computer vision. Il montre une progression structuree : partir d'un premier projet complet sur la leucemie, puis etendre la meme rigueur a d'autres datasets publics lies au cancer.

Le projet ne pretend pas resoudre un probleme medical reel. Il sert a demontrer des competences ML : preparation de donnees, entrainement PyTorch, evaluation, interpretabilite, benchmark, analyse d'erreurs, documentation et tests.

## Projet Termine : Leukemia

Le sous-projet [Leukemia](../projects/leukemia/README.md) est complet.

Points forts :

- dataset public Kaggle ;
- classification `normal` vs `leukemia_blast` ;
- PyTorch / Torchvision ;
- Streamlit ;
- Grad-CAM ;
- benchmark ResNet18 / MobileNetV3 / EfficientNet ;
- analyse false positives / false negatives ;
- tests automatises ;
- README avec captures et commandes de reproduction.

Resultats principaux :

| Version | Resultat |
| --- | --- |
| V1 | 91.69 % test accuracy avec ResNet18 |
| V2.1 | benchmark reel de trois architectures |
| V2.2 | 1601 images test analysees, 133 erreurs, 63 false positives, 70 false negatives |

## Projets Suivants Et En Cours

| Projet | Statut | Objectif |
| --- | --- | --- |
| Lung + Colon | V1 initialisee | Pipeline LC25000 5 classes, dataset reel a lancer localement |
| Breast | prevu | Classification benin / malin sur dataset public |
| Metastasis | prevu | Detection ou classification de patches |
| MultiCancer | prevu | Synthese portfolio et comparaison transversale |

## Competences Demontrees

- Structuration d'un projet Python.
- Pipelines de computer vision.
- Transfer learning.
- Evaluation avec metriques de classification.
- Interpretation precision / recall / F1.
- Analyse d'erreurs.
- Grad-CAM.
- Streamlit pour demo locale.
- Tests automatises avec Pytest.
- Documentation portfolio et communication technique.

## Positionnement

R.C.C.I.A est utile pour un CV, un portfolio GitHub, un entretien ou un post LinkedIn. Le bon angle est :

> Projet IA/data educatif montrant une pipeline ML complete et une demarche d'evaluation rigoureuse sur datasets publics.

Le mauvais angle a eviter :

> Outil de detection medicale ou aide au diagnostic.

## Disclaimer

Tous les sous-projets R.C.C.I.A sont des demonstrateurs educatifs. Ils ne fournissent pas de diagnostic medical, ne remplacent pas un professionnel de sante et ne doivent pas etre utilises pour prendre ou orienter une decision medicale.
