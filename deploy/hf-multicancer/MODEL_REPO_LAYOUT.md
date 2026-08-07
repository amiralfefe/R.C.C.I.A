# Model Repository Layout

Le repository Hugging Face Model reference par `HF_MODEL_REPO_ID` doit contenir
exactement les chemins distants suivants :

```text
leukemia/best_model.pt
breast/efficientnet_b0/best_model.pt
metastasis/efficientnet_b0/best_model.pt
lung_colon/multiclass/efficientnet_b0/best_model.pt
lung_colon/binary/resnet18/best_model.pt
```

## Correspondance Verifiee

| Parcours | Chemin distant | Chemin local attendu par l'adaptateur | Taille locale observee |
| --- | --- | --- | ---: |
| Leukemia | `leukemia/best_model.pt` | `projects/leukemia/outputs/best_model.pt` | 42.71 MiB |
| Breast | `breast/efficientnet_b0/best_model.pt` | `projects/breast/outputs/model_comparison/efficientnet_b0/best_model.pt` | 15.58 MiB |
| Metastasis | `metastasis/efficientnet_b0/best_model.pt` | `projects/metastasis/outputs/model_comparison/efficientnet_b0/best_model.pt` | 15.58 MiB |
| LungColon multiclass | `lung_colon/multiclass/efficientnet_b0/best_model.pt` | `projects/lung_colon/outputs/model_comparison/efficientnet_b0/best_model.pt` | 15.59 MiB |
| LungColon binary | `lung_colon/binary/resnet18/best_model.pt` | `projects/lung_colon/outputs/binary_resnet18/best_model.pt` | 42.71 MiB |

Total local observe : environ **132.17 MiB**.

Les noms et chemins locaux ont ete verifies contre les adaptateurs MultiCancer de la
release `multicancer-v1`. Les fichiers restent exclus de Git. Leur creation ou upload
vers Hugging Face doit etre effectue manuellement et n'est pas realise par ce bundle.
