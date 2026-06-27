# Setup Windows

Ce projet est pense pour etre lance localement sur Windows avec un environnement virtuel Python.

## Option rapide

Depuis `C:\VSCODE\R.C.C.I.A` :

```bat
scripts\setup_windows.bat
```

Le script cree `.venv`, installe les dependances et lance les tests.

## Installation manuelle

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m pytest
```

## Commandes utiles

Preparer les splits :

```powershell
python scripts/split_image_folder.py --input data/raw --output data/processed --val-ratio 0.15 --test-ratio 0.15
```

Entrainer :

```powershell
python -m cancer_cell_vision.train --data-dir data/processed --epochs 5 --batch-size 16 --output-dir outputs
```

Evaluer :

```powershell
python -m cancer_cell_vision.evaluate --data-dir data/processed --checkpoint outputs/best_model.pt --output-dir outputs/eval
```

Predire sur une image :

```powershell
python -m cancer_cell_vision.predict --checkpoint outputs/best_model.pt --image data/processed/test/normal/example.jpg --pretty
```

Lancer l'app :

```powershell
streamlit run app.py
```
