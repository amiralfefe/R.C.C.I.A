# MultiCancer - Project Summary

## Resume Court

MultiCancer est le hub de synthese du monorepo R.C.C.I.A. Il reunit quatre projets de
computer vision appliques a des datasets publics lies au cancer, sans fusionner leurs
datasets ni construire un modele universel artificiel.

Le projet route explicitement l'utilisateur vers un pipeline specialise, charge un seul
checkpoint a la demande et normalise uniquement la presentation des predictions,
probabilites, limites et visualisations Grad-CAM.

MultiCancer est un demonstrateur educatif et portfolio. Ce n'est pas un dispositif
medical, un outil de diagnostic, une validation clinique ou une aide a la decision de
sante.

## Vision Du Hub

Les quatre projets utilisent des modalites, classes, resolutions, architectures,
protocoles de split et metriques differents. Les homogeniser dans un modele unique
masquerait ces differences. MultiCancer conserve donc les pipelines specialises et leur
contexte methodologique.

Le hub ne choisit jamais automatiquement un cancer ou une modalite a partir de l'image.
Le projet et, pour LungColon, le mode sont toujours selectionnes explicitement.

## Projets Integres

| Parcours | Dataset | Modele local | Resolution | Sortie |
| --- | --- | --- | ---: | --- |
| Leukemia | Leukemia Classification | ResNet18 | 224 | `normal`, `leukemia_blast` |
| Breast | BreakHis | EfficientNet-B0 | 224 | `benign`, `malignant` |
| Metastasis | PCam subset | EfficientNet-B0 | 96 | `non_metastatic`, `metastatic` |
| LungColon multiclass | LC25000 | EfficientNet-B0 | 224 | cinq classes histologiques |
| LungColon binary | LC25000 | ResNet18 | 224 | `benign`, `malignant` |

Les checkpoints, datasets et outputs restent locaux et ne sont pas versionnes.

## Architecture

```text
Streamlit
  -> registre de projets
  -> ModelManager
  -> Adapter specialise
  -> pipeline existant + checkpoint local
  -> PredictionResult commun
  -> probabilites + Grad-CAM + limites
```

### Contrat Adapter

Chaque adaptateur expose le meme cycle de vie :

- `metadata()` pour le contexte stable ;
- `checkpoint_status()` sans charger le modele ;
- `load()` pour le chargement paresseux et idempotent ;
- `predict()` pour une sortie normalisee ;
- `explain()` pour Grad-CAM lorsque disponible ;
- `unload()` pour liberer le modele et le cache accelerateur applicable.

### ModelManager Et Lazy Loading

`ModelManager` garantit un seul adaptateur actif. Avant tout changement de projet, il
decharge le modele precedent. Un checkpoint absent produit un etat controle et laisse le
hub consultable.

### PredictionResult

Le schema commun contient la classe predite, l'index, la confiance, les probabilites,
le modele, la resolution, les avertissements et des metadonnees optionnelles. Il
normalise l'affichage sans modifier la logique des projets specialises.

### ThresholdDecision Metastasis

La prediction Metastasis conserve l'argmax et les probabilites brutes. Le slider derive
une `ThresholdDecision` separee, sans recharger le modele ni relancer l'inference. Cette
exploration illustre le compromis faux positifs / faux negatifs ; elle ne recommande
aucun seuil medical.

### Modes LungColon

LungColon impose un choix explicite entre :

- le checkpoint EfficientNet-B0 cinq classes ;
- le checkpoint ResNet18 binaire.

Une bascule de mode decharge le modele actif et efface prediction, image, probabilites et
Grad-CAM precedents avant tout nouveau chargement.

## Interpretabilite

Les cinq parcours produisent une visualisation Grad-CAM en memoire lorsque le checkpoint
local est disponible. Grad-CAM aide a explorer les zones influencant la prediction, mais
ne constitue ni une explication causale ni une preuve medicale.

## QA Finale V1

La QA locale finale a valide les cinq parcours avec les vrais checkpoints :

- somme des probabilites egale a 1 a tolerance numerique pres ;
- Grad-CAM `224x224x3` pour Leukemia, Breast et les deux modes LungColon ;
- Grad-CAM `96x96x3` pour Metastasis ;
- unload a chaque changement de projet ou de mode ;
- `PredictionResult` Metastasis inchange lors du mouvement du seuil ;
- retour final vers Leukemia sans modele precedent residuel ;
- aucune exception dans le scenario Streamlit AppTest complet.

Un exemple Breast `malignant` a ete predit `benign` avec une confiance elevee. Ce cas est
conserve car il montre qu'une confiance forte ne garantit pas une prediction correcte et
renforce l'importance de l'analyse d'erreurs patient-aware.

La suite monorepo finale est executee avec :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Resultat final local : **120 tests passes**. Les six controles AppTest reels restent
CI-safe grace a un skip explicite lorsque checkpoints ou images locales sont absents.

## Gestion Des Erreurs

Les tests couvrent notamment : checkpoint absent ou incompatible, image invalide,
projet inconnu, mode LungColon invalide, seuil invalide, Grad-CAM indisponible et unload
idempotent. Ces situations produisent une erreur controlee plutot qu'un crash du hub.

## Limites

- Les resultats proviennent de datasets publics et de protocoles differents.
- Les accuracies ne doivent pas servir a classer directement les projets entre eux.
- BreakHis montre une variabilite patient importante malgre un split patient-aware.
- LC25000 peut produire des scores tres eleves sur un protocole local relativement
  facile.
- Le subset PCam ne couvre pas le corpus complet.
- Le projet Leukemia ne documente pas de split patient-aware.
- Grad-CAM reste exploratoire.
- Aucun projet n'a de validation externe, multi-centres ou clinique.

## Statut Final

Statut : **MultiCancer V1 complete / portfolio-ready**.

Tag de reference : `multicancer-v1`.

Le hub presente quatre pipelines specialises et cinq parcours explicites. Il ne doit
jamais etre presente comme une IA capable de diagnostiquer plusieurs cancers.
