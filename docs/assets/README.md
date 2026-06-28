# Captures Portfolio

Ce dossier est reserve aux captures d'ecran de la demo Streamlit.

Ne pas y placer de donnees medicales sensibles, de tokens, de checkpoints ou d'exports complets du dataset.

## Captures Recommandees

1. `streamlit-home.png`
   - Page d'accueil Streamlit.
   - Sidebar visible.
   - Etat du modele : `Modele charge`.
   - Disclaimer medical visible.
   - Section `Resultats V1` visible.

2. `prediction-leukemia-gradcam.png`
   - Image de test `leukemia_blast` chargee.
   - Classe predite visible.
   - Confiance visible.
   - Grad-CAM visible.
   - Message de disclaimer encore accessible.

3. `prediction-normal.png`
   - Image de test `normal` chargee.
   - Classe predite visible.
   - Tableau ou graphique des probabilites visible.
   - Section limites ou disclaimer accessible dans la page.

## Commande De Demo

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Images de test locales conseillees :

```text
data\processed\test\leukemia_blast\leukemia_blast_000004.bmp
data\processed\test\normal\normal_000001.bmp
```

## Integration README

Quand les captures sont ajoutees, elles peuvent etre referencees dans `README.md` avec :

```markdown
![Streamlit modele charge](docs/assets/streamlit-home.png)
![Prediction leukemia_blast avec Grad-CAM](docs/assets/prediction-leukemia-gradcam.png)
![Prediction normal avec probabilites](docs/assets/prediction-normal.png)
```

Avant de commit, verifier que seuls `README.md`, `docs/` ou d'autres fichiers source/doc sont suivis par Git.
