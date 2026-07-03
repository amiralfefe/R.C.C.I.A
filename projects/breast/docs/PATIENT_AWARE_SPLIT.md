# Patient-Aware Split

## Pourquoi C'est Important

Dans un dataset histopathologique, un meme patient peut avoir plusieurs images ou patches. Si des images du meme patient se retrouvent a la fois dans `train` et dans `test`, le modele peut apprendre des indices propres au patient ou a la preparation des lames.

Cela peut produire des scores trop optimistes.

## Limite D'un Split Random Par Image

Un split random classique separe les images individuellement. Il est simple et utile pour demarrer, mais il ne garantit pas que les patients soient separes.

Exemple de risque :

```text
patient_001/image_a.png -> train
patient_001/image_b.png -> test
```

Dans ce cas, le test n'est pas totalement independant.

## Principe Patient-Aware

Un split patient-aware groupe les images par `patient_id`, puis affecte chaque patient a un seul split :

```text
patient_001 -> train
patient_002 -> val
patient_003 -> test
```

Le but est de reduire le data leakage entre train, validation et test.

## Extraction Patient Dans Ce Projet

`prepare_breakhis_dataset.py` tente d'extraire `patient_id` depuis :

- les noms de fichiers BreakHis du type `SOB_B_A-14-22549AB-40-001.png` ;
- les chemins contenant `patient`, `case` ou `pt`.

Pour un nom BreakHis comme :

```text
SOB_B_A-14-22549AB-40-001.png
```

le projet tente d'extraire :

```text
patient_id = 14-22549AB
magnification = 40X
```

## Commande

```powershell
.\.venv\Scripts\python.exe projects\breast\scripts\split_image_folder.py --input projects\breast\data\raw --output projects\breast\data\processed --metadata projects\breast\data\raw\metadata.csv --patient-aware --val-ratio 0.15 --test-ratio 0.15
```

## Si Patient ID N'est Pas Disponible

Ne pas presenter le split comme patient-aware.

Options propres :

1. Utiliser le split classique et documenter la limite.
2. Fournir un `metadata.csv` avec une colonne `patient_id` complete.
3. Chercher une version du dataset avec metadata patient exploitable.

Le script refuse le mode `--patient-aware` si un `patient_id` manque dans `metadata.csv`.

## Interpretation Portfolio

Le bon discours est :

> J'ai prepare le pipeline pour supporter un split patient-aware afin de limiter le risque de data leakage. Si les identifiants patients ne sont pas disponibles, le projet documente explicitement cette limite au lieu de presenter un split random comme equivalent.

## Disclaimer

Cette demarche ameliore la rigueur experimentale d'un projet educatif, mais elle ne constitue pas une validation clinique.
