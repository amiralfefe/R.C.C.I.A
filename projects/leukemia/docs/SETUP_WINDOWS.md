# Setup Windows

Ce sous-projet est pense pour etre lance localement sur Windows avec un environnement virtuel Python cree a la racine du monorepo.

## Option Rapide

Depuis la racine de votre clone R.C.C.I.A :

```bat
projects\leukemia\scripts\setup_windows.bat
```

Le script cree `.venv` a la racine, installe les dependances et lance les tests.

## Installation Manuelle

Depuis la racine de votre clone R.C.C.I.A (Python 3.11 recommande) :

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
cd projects\leukemia
```

## Commandes Utiles

Les commandes suivantes supposent d'etre dans `projects\leukemia`.

Preparer les splits :

```powershell
..\..\.venv\Scripts\python.exe scripts\split_image_folder.py --input data/raw --output data/processed --val-ratio 0.15 --test-ratio 0.15
```

Entrainer :

```powershell
..\..\.venv\Scripts\python.exe -m rccia_leukemia.train --data-dir data/processed --epochs 5 --batch-size 16 --output-dir outputs
```

Evaluer :

```powershell
..\..\.venv\Scripts\python.exe -m rccia_leukemia.evaluate --data-dir data/processed --checkpoint outputs/best_model.pt --output-dir outputs/eval
```

Predire sur une image :

```powershell
..\..\.venv\Scripts\python.exe -m rccia_leukemia.predict --checkpoint outputs/best_model.pt --image data/processed/test/normal/example.jpg --pretty
```

Lancer l'app :

```powershell
..\..\.venv\Scripts\streamlit.exe run app.py
```
