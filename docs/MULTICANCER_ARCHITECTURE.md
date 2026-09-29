# MultiCancer - architecture livree

Etat documentaire : 29 septembre 2026. Cette page remplace la description cible
V0 ; le [scope initial](MULTICANCER_SCOPE.md) reste un document historique.
Le hub integre quatre pipelines et cinq parcours, sans detecter automatiquement
la modalite ni construire de modele medical universel.

## Flux d'une session

```text
Choix explicite du projet (et du mode LungColon)
  -> ModelManager : unload avant changement
  -> BaseAdapter specialise : statut du checkpoint
  -> upload PNG/JPEG : decodeur borne rccia_common
  -> load a la demande + preprocessing du pipeline
  -> PredictionResult : classe, probabilites, modele, limites
  -> ExplanationResult : Grad-CAM optionnel
  -> Metastasis : ThresholdDecision derivee sans nouvelle inference
```

## Composants reels

| Composant | Role |
| --- | --- |
| `projects/multicancer/app.py` | Interface, session, affichage des resultats et erreurs |
| `multicancer/registry.py` | Metadonnees, resultats historiques et limites des projets |
| `multicancer/schemas.py` | Dataclasses des metadonnees, checkpoints, predictions, explications et seuils |
| `multicancer/model_manager.py` | Selection explicite et cycle de vie d'un adaptateur par session |
| `multicancer/adapters/base.py` | Contrat commun : metadata, checkpoint_status, load, predict, explain, unload |
| `multicancer/adapters/*_adapter.py` | Quatre adaptateurs deleguant aux pipelines specialises |
| `multicancer/thresholds.py` | Decision exploratoire separee de l'argmax original |
| `projects/rccia_common/` | Validation des uploads et des chemins de preparation |
| `deploy/streamlit-multicancer/app.py` | Bootstrap optionnel des poids avant lancement du hub |

Les chemins `multicancer/...` sont relatifs a `projects/multicancer/`.
Il n'existe pas de service REST ou de base de donnees. `router.py` et un package
`ui/` figuraient dans la cible V0 : ce ne sont pas des composants livres ni des
prerequis a ajouter. Le routage est assure par le ModelManager et l'interface.

## Poids et contrats

Chaque pipeline utilise un checkpoint contenant l'etat du modele, les classes,
la resolution et le nom d'architecture. L'adaptateur conserve le preprocessing
du pipeline source ; le hub normalise la presentation, pas les datasets.
Voir les [chemins de checkpoints](LOCAL_SETUP.md#inference-avec-checkpoints).

LungColon impose un mode `multiclass` (EfficientNet-B0) ou `binary` (ResNet18).
Changer ce mode libere le modele actif. Metastasis conserve une PredictionResult
independante du seuil : modifier le slider ne remplace pas son argmax et ne
relance pas l'inference.

## Memoire et erreurs

- Un modele au plus par session, **pas un quota global pour tous les utilisateurs**.
- Chargement paresseux ; metadonnees et interface disponibles sans poids.
- Nouvelle image ou nouvelle analyse : ancien resultat et Grad-CAM invalides.
- Grad-CAM reussi mis en cache pour la prediction courante, pas recalcule par le slider.
- Checkpoint manquant/incompatible, image invalide et explication indisponible
  produisent des messages controles.
- Les uploads du hub ne sont pas enregistres dans les datasets ou reutilises pour
  entrainement ; voir la [notice de confidentialite](PRIVACY.md).

## Frontiere de deploiement

Le lancement direct du hub ne telecharge rien. Le bootstrap Cloud telecharge les
cinq poids vers des chemins fixes depuis un depot HF prive, avec un token de
lecture cote serveur. Il ne charge pas les modeles en RAM. L'installation locale
utilise des temporaires distincts et un verrou de remplacement par processus.
Ces corrections ne sont effectives a distance qu'apres publication autorisee.

Le Dockerfile HF historique clone le tag `multicancer-v1`, donc pas les correctifs
posterieurs. Il n'est pas le chemin recommande pour essayer la version locale.
La [validation locale](PORTFOLIO_VALIDATION.md) ne certifie ni le service Cloud,
ni une charge multiutilisateur, ni une aptitude clinique.
