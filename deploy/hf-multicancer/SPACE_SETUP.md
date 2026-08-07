# Hugging Face Space Setup

Cette procedure publie MultiCancer sous forme de Docker Space sans inclure les
checkpoints dans GitHub ou dans le repository du Space.

## 1. Prerequis

- creer manuellement un repository Hugging Face de type **Model** ;
- y placer les cinq fichiers decrits dans `MODEL_REPO_LAYOUT.md` ;
- conserver ce repository prive si souhaite ;
- creer un token Hugging Face **read-only** pouvant lire ce repository.

Ce bundle ne cree aucun repository distant et n'upload aucun checkpoint.

## 2. Creer Le Space

1. Creer un nouveau Space nomme `rccia-multicancer`.
2. Selectionner le SDK **Docker**.
3. Choisir une visibilite publique pour la demonstration portfolio.
4. Copier le contenu de `deploy/hf-multicancer/` a la racine du repository du Space.

Le Dockerfile clone directement la release GitHub `multicancer-v1` et verifie le commit
`a51772da5d50eb8d758c01d737ea09f2f388cea6`. Le monorepo complet n'a donc pas besoin
d'etre duplique dans le repository du Space.

## 3. Configurer Variables Et Secret

Dans **Settings > Variables and secrets** du Space :

- variable `HF_MODEL_REPO_ID` : `<compte>/<repository-modeles>` ;
- secret `HF_TOKEN` : token Hugging Face read-only.

Ne jamais placer la valeur reelle du token dans Git, une capture ou un log. Pour un
repository de modeles public, le token peut etre omis ; pour un repository prive, son
absence produit une erreur de demarrage explicite.

## 4. Build Local Optionnel

Le contexte Docker est volontairement limite au bundle de deploiement :

```powershell
docker build -t rccia-multicancer deploy/hf-multicancer
```

Pour un repository de modeles public :

```powershell
docker run --rm -p 7860:7860 `
  -e HF_MODEL_REPO_ID=<compte>/<repository-modeles> `
  rccia-multicancer
```

Pour un repository prive, definir `HF_TOKEN` dans la session locale puis transmettre
la variable sans ecrire sa valeur dans la commande :

```powershell
$env:HF_TOKEN = "<token-read-only>"
docker run --rm -p 7860:7860 `
  -e HF_MODEL_REPO_ID=<compte>/<repository-modeles> `
  -e HF_TOKEN `
  rccia-multicancer
```

Supprimer ensuite la variable de la session :

```powershell
Remove-Item Env:HF_TOKEN
```

## 5. Comportement Au Demarrage

1. `download_models.py` telecharge uniquement les checkpoints absents sur le disque.
2. Aucun modele PyTorch n'est charge pendant cette etape.
3. `entrypoint.py` lance Streamlit sur `0.0.0.0:7860`.
4. Le `ModelManager` existant charge ensuite uniquement le parcours choisi et decharge
   le modele actif lors d'une bascule.

Un cold start peut etre long : environ 132 MiB de checkpoints doivent etre presents sur
le disque et l'inference CPU peut etre lente. Le stockage du Space est ephemere ; un
redemarrage peut donc imposer un nouveau telechargement.

## 6. Confidentialite Et Limites

- aucune image utilisateur n'est stockee durablement par l'application ;
- le Space ne remplace pas une politique de confidentialite ou une revue de securite ;
- les checkpoints et datasets restent hors Git ;
- le Space est un demonstrateur educatif / portfolio, sans diagnostic ni validation
  clinique ;
- les probabilites, Grad-CAM et seuils ne doivent orienter aucune decision de sante.
