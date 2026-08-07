# MultiCancer - Interview Pitch

## Pitch 30 Secondes

J'ai construit MultiCancer, le hub final de mon monorepo R.C.C.I.A. Il regroupe quatre
pipelines de computer vision specialises : Leukemia, Breast, Metastasis et LungColon.
Plutot que de fabriquer un modele universel, j'ai cree une architecture d'adaptateurs
avec routage explicite, lazy loading, un seul checkpoint en memoire, sortie commune,
probabilites et Grad-CAM. Le hub gere aussi un seuil exploratoire separe pour Metastasis
et deux checkpoints explicites pour LungColon. C'est un projet educatif et portfolio,
pas un outil medical.

## Pitch 2 Minutes

R.C.C.I.A contient quatre projets specialises construits sur des datasets publics lies
au cancer. Ils n'ont ni les memes images, ni les memes classes, ni les memes resolutions,
ni les memes protocoles. MultiCancer est le hub qui permet de les presenter dans une
seule interface sans gommer ces differences.

J'ai defini un contrat `BaseAdapter` commun pour les metadonnees, le statut checkpoint,
le chargement, la prediction, Grad-CAM et l'unload. Un `ModelManager` garantit qu'un seul
adaptateur peut conserver un modele charge. Le chargement est paresseux : ouvrir le hub
ou changer de projet ne charge aucun checkpoint tant qu'une prediction n'est pas lancee.

Le resultat brut est normalise dans un `PredictionResult`, mais chaque adaptateur
continue a utiliser son pipeline specialise. Metastasis ajoute une `ThresholdDecision`
separee : changer le seuil ne modifie pas l'argmax, ne recharge pas le modele et ne
relance pas l'inference. LungColon gere deux modes explicites avec deux architectures et
deux checkpoints ; une bascule de mode decharge le precedent et efface toute sortie
obsolette.

La QA finale a teste les cinq parcours avec les vrais checkpoints, les probabilites,
Grad-CAM, les changements de projet, les erreurs controlees et le cycle memoire. Le cas
Breast montre aussi qu'une forte confiance peut accompagner une erreur, d'ou
l'importance du split patient-aware et de l'analyse d'erreurs.

Je presente ce projet comme une synthese d'architecture ML et de rigueur methodologique,
jamais comme un dispositif medical ou une IA universelle de diagnostic.

## Points Techniques A Expliquer

- Architecture par adaptateurs specialises.
- Registre de metadonnees independant des checkpoints.
- Lazy loading et unload idempotent.
- Garantie d'un seul modele actif via `ModelManager`.
- Schema commun `PredictionResult` sans uniformiser les taches.
- Gestion de ResNet18 et EfficientNet-B0 dans le meme hub.
- Preprocessing 96 ou 224 selon le pipeline.
- Deux modes LungColon, deux checkpoints, aucun chargement simultane.
- Separation argmax / `ThresholdDecision` pour Metastasis.
- Split patient-aware et analyse par grossissement pour Breast.
- Prudence sur les scores LC25000.
- ROC-AUC, PR-AUC et compromis FP/FN pour Metastasis.
- Grad-CAM et ses limites.
- Erreurs controlees et QA AppTest.
- Exclusion des datasets, checkpoints, outputs et tokens de Git.

## Questions Recruteur

### Pourquoi ne pas utiliser un modele universel ?

Les datasets, modalites, classes, resolutions et protocoles sont heterogenes. Un modele
universel aurait masque ces differences sans protocole d'entrainement transversal
valide. Le hub conserve donc la specialisation et normalise seulement l'interface.

### Pourquoi une architecture Adapter ?

Elle donne un contrat stable a l'interface tout en laissant chaque projet conserver son
preprocessing, son modele, ses classes et sa logique Grad-CAM.

### Pourquoi le lazy loading ?

Les checkpoints sont lourds et independants. Le chargement a la demande reduit la
memoire utilisee et evite de charger quatre modeles pour consulter une seule page.

### Pourquoi un ModelManager ?

Il centralise le cycle de vie et garantit qu'un changement de projet decharge d'abord le
modele precedent. Cette regle est plus simple a tester qu'une gestion dispersee dans
l'interface.

### Pourquoi un PredictionResult commun ?

Il fournit a Streamlit une sortie previsible : classe, confiance, probabilites, modele,
resolution et avertissements. Il ne change pas l'inference specialisee.

### Comment gerez-vous plusieurs architectures ?

Chaque checkpoint decrit ses classes, son architecture et sa resolution. L'adaptateur
delegue au package specialise et verifie la compatibilite avant d'exposer la sortie
commune.

### Comment gerez-vous les deux modes LungColon ?

Le mode est choisi explicitement. Chaque mode a ses propres classes, architecture,
resolution et checkpoint. Une bascule appelle `unload()` et efface l'etat de prediction
avant de charger le nouveau modele.

### Quelle difference entre argmax et decision au seuil ?

L'argmax est la classe ayant la probabilite la plus haute. La decision au seuil applique
une regle distincte a la probabilite `metastatic`. Elle peut differer de l'argmax sans
modifier la prediction brute.

### Pourquoi le split patient-aware Breast est important ?

Il empeche les images d'un meme patient d'etre reparties entre entrainement et test, ce
qui reduit un risque de fuite de donnees. L'analyse montre toutefois qu'un patient peut
concentrer beaucoup d'erreurs.

### Pourquoi rester prudent avec LC25000 ?

Le dataset peut etre relativement facile dans certains protocoles et produire des scores
tres eleves. Ces scores locaux ne prouvent ni generalisation externe ni performance
clinique.

### Pourquoi ROC-AUC et PR-AUC pour Metastasis ?

Ces metriques evaluent la separation des classes et le compromis precision/recall au-dela
d'un seuil unique. Elles completent l'accuracy et l'analyse FP/FN.

### Pourquoi Grad-CAM ?

Grad-CAM aide a inspecter les zones qui influencent une prediction. Il reste exploratoire,
ne prouve pas une causalite et ne valide pas medicalement le modele.

### Comment gerez-vous un checkpoint absent ?

Le statut est affiche avant chargement. Le hub reste consultable et retourne un message
controle au lieu de planter ou de telecharger automatiquement un modele.

### Pourquoi ce n'est pas un outil medical ?

Les modeles n'ont pas de validation clinique, externe ou multi-centres, ne sont pas
certifies et ne doivent orienter aucune decision de sante. Le but est pedagogique et
portfolio.
