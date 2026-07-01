# V2 - Model Comparison + Evaluation Reports

## Objectif

La V2 ajoute un workflow de comparaison de modeles pour renforcer la credibilite ML du projet R.C.C.I.A sans changer le cadrage : demonstrateur educatif IA/data, pas outil medical et pas diagnostic.

Elle permet de comparer plusieurs architectures sur le meme dataset prepare dans `data/processed/`, avec les memes splits `train`, `val` et `test`.

## Modeles Supportes

- `resnet18`
- `efficientnet_b0`
- `mobilenet_v3_small`

`resnet18` reste le modele par defaut pour conserver la compatibilite avec la V1.

## Entrainer Un Modele

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.train --data-dir data\processed --model resnet18 --epochs 5 --batch-size 16 --output-dir outputs\resnet18
```

Autres exemples :

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.train --data-dir data\processed --model efficientnet_b0 --epochs 5 --batch-size 16 --output-dir outputs\efficientnet_b0
.\.venv\Scripts\python.exe -m cancer_cell_vision.train --data-dir data\processed --model mobilenet_v3_small --epochs 5 --batch-size 16 --output-dir outputs\mobilenet_v3_small
```

Par defaut, les modeles utilisent les poids ImageNet si disponibles. Pour un smoke test rapide sans telechargement de poids :

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.train --data-dir data\processed --model mobilenet_v3_small --epochs 1 --batch-size 16 --output-dir outputs\mobilenet_smoke --no-pretrained
```

## Comparer Plusieurs Modeles

Commande complete souhaitee :

```powershell
.\.venv\Scripts\python.exe scripts\run_model_comparison.py --data-dir data\processed --epochs 5 --batch-size 16 --output-dir outputs\model_comparison
```

Commande courte CPU possible :

```powershell
.\.venv\Scripts\python.exe scripts\run_model_comparison.py --data-dir data\processed --epochs 2 --batch-size 16 --output-dir outputs\model_comparison_smoke --no-pretrained
```

## Smoke Run Local

Un smoke run CPU a ete lance pour valider le workflow sans telechargement de poids ImageNet :

```powershell
.\.venv\Scripts\python.exe scripts\run_model_comparison.py --data-dir data\processed --models resnet18 mobilenet_v3_small --epochs 1 --batch-size 64 --image-size 64 --output-dir outputs\model_comparison_smoke --no-pretrained
```

Resultats obtenus :

| Modele | Accuracy | F1 normal | F1 leukemia_blast | Train time |
| --- | ---: | ---: | ---: | ---: |
| `resnet18` | 0.8164 | 0.6811 | 0.8711 | 42.85 s |
| `mobilenet_v3_small` | 0.6821 | 0.0000 | 0.8110 | 28.49 s |

Important : ce smoke run sert seulement a verifier le workflow de comparaison et la generation des rapports. Il utilise 1 epoch, des images 64x64 et `--no-pretrained`. Il ne remplace pas une comparaison V2 complete avec les trois modeles, image size 224, epochs 5 et poids pre-entraines.

## Benchmark Reel V2.1

La V2.1 a ete lancee avec un protocole plus proche d'une comparaison exploitable tout en restant raisonnable sur CPU :

- poids ImageNet pre-entraines ;
- 3 epochs ;
- batch size 16 ;
- image size 224 ;
- meme dossier `data/processed` et memes splits `train` / `val` / `test`.

Commandes lancees :

```powershell
.\.venv\Scripts\python.exe scripts\run_model_comparison.py --data-dir data\processed --models resnet18 mobilenet_v3_small --epochs 3 --batch-size 16 --image-size 224 --output-dir outputs\model_comparison_real
.\.venv\Scripts\python.exe scripts\run_model_comparison.py --data-dir data\processed --models efficientnet_b0 --epochs 3 --batch-size 16 --image-size 224 --output-dir outputs\model_comparison_real_efficientnet
```

Resultats obtenus sur le test set :

| Modele | Accuracy | Precision normal | Recall normal | F1 normal | Precision leukemia_blast | Recall leukemia_blast | F1 leukemia_blast | Train time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `resnet18` | 0.8988 | 0.9161 | 0.7505 | 0.8251 | 0.8927 | 0.9679 | 0.9288 | 586.95 s |
| `mobilenet_v3_small` | 0.8976 | 0.8808 | 0.7839 | 0.8295 | 0.9042 | 0.9505 | 0.9268 | 277.76 s |
| `efficientnet_b0` | 0.8863 | 0.8724 | 0.7525 | 0.8080 | 0.8916 | 0.9487 | 0.9193 | 890.49 s |

Lecture :

- `resnet18` garde la meilleure accuracy et le meilleur recall `leukemia_blast` sur ce benchmark court.
- `mobilenet_v3_small` est tres proche en accuracy, obtient le meilleur F1 `normal` et entraine beaucoup plus vite sur CPU.
- `efficientnet_b0` est le plus lent ici et ne depasse pas les deux autres modeles avec 3 epochs.

La baseline V1 reste separee : elle correspond a un run ResNet18 5 epochs documente dans le README. La V2 smoke test reste separee aussi : elle valide seulement l'orchestration. La V2.1 est le premier benchmark reel comparable entre architectures, mais elle reste experimentale et non medicale.

Fichiers generes :

```text
outputs/model_comparison_real/
|-- summary.csv
|-- summary.json
|-- resnet18/
|-- mobilenet_v3_small/

outputs/model_comparison_real_efficientnet/
|-- summary.csv
|-- summary.json
`-- efficientnet_b0/
```

Chaque sous-dossier modele contient le checkpoint local, les courbes train/val, le rapport de classification JSON/CSV et la matrice de confusion PNG. Ces fichiers restent dans `outputs/` et ne doivent pas etre committes.

## Fichiers Generes

Le workflow ecrit dans `outputs/`, qui reste ignore par Git :

```text
outputs/model_comparison/
|-- summary.csv
|-- summary.json
|-- resnet18/
|   |-- best_model.pt
|   |-- train/
|   |   |-- training_setup.json
|   |   |-- training_history.json
|   |   |-- training_history.csv
|   |   |-- training_loss.png
|   |   `-- training_accuracy.png
|   `-- eval/
|       |-- test_classification_report.json
|       |-- test_classification_report.csv
|       `-- test_confusion_matrix.png
|-- efficientnet_b0/
`-- mobilenet_v3_small/
```

## Colonnes Du Summary

- `model`
- `accuracy`
- `precision_normal`
- `recall_normal`
- `f1_normal`
- `precision_leukemia_blast`
- `recall_leukemia_blast`
- `f1_leukemia_blast`
- `checkpoint_path`
- `train_time_seconds`

## Interpreter Les Metriques

- Accuracy : proportion globale de predictions correctes.
- Precision : parmi les images predites dans une classe, proportion correcte.
- Recall : parmi les images reelles d'une classe, proportion retrouvee par le modele.
- F1-score : moyenne harmonique entre precision et recall.

Dans ce projet, le recall sur `leukemia_blast` est important pour analyser les erreurs experimentales. Cela ne transforme pas le modele en outil medical : les scores restent des indicateurs de projet portfolio.

## Limites

- Dataset public Kaggle, sans validation clinique independante.
- Classes desequilibrees entre `normal` et `leukemia_blast`.
- Resultats dependants du split, du preprocessing et des hyperparametres.
- Une comparaison courte CPU sert seulement de smoke test ou d'indication preliminaire.
- Aucun resultat ne doit etre interprete comme une validation medicale.

## Versionner Proprement

Ne pas commit :

- `data/raw/`
- `data/processed/`
- `outputs/`
- checkpoints `*.pt`, `*.pth`, `*.ckpt`
- token Kaggle, `access_token`, `kaggle.json`
- `.venv/`
