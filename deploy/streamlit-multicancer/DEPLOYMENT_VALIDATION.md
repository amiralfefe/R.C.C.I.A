# Streamlit MultiCancer Deployment Validation

Validation date: 2026-08-07

Public URL: https://rccia-multicancer.streamlit.app/

## Public Status

| Check | Status |
| --- | --- |
| Anonymous public access | pass after changing Streamlit sharing from restricted to public |
| Private Hugging Face repository access | pass, no 401/403 or checkpoint download error |
| Deployment bootstrap | pass |
| Expected checkpoints | 5/5 detected by the public application |
| Lazy loading | pass, welcome screen keeps the selected model out of memory |
| Single-model lifecycle | pass across project and LungColon mode changes |
| Leukemia | pass |
| Breast | pass |
| Metastasis | pass |
| LungColon multiclass | pass |
| LungColon binary | pass |
| Grad-CAM | pass on all five prediction paths |
| Controlled invalid states | pass |
| Reboot to ready state | approximately 97 seconds |
| Mobile viewport | not independently emulated in the available browser environment |

The first anonymous request was redirected to a Streamlit access-denied page because
the application setting was `Only specific people can view this app`. The setting was
changed to `This app is public and searchable`. A fresh anonymous browser session then
loaded the application without authentication.

No `Checkpoint download failed` message, Hugging Face `401`/`403`, Streamlit exception
or Python traceback was observed during the public validation. Streamlit logs did not
expose a detailed line for each checkpoint download. The public UI nevertheless
detected all five expected local paths, and every corresponding model loaded and ran:

- `projects/leukemia/outputs/best_model.pt`
- `projects/breast/outputs/model_comparison/efficientnet_b0/best_model.pt`
- `projects/metastasis/outputs/model_comparison/efficientnet_b0/best_model.pt`
- `projects/lung_colon/outputs/model_comparison/efficientnet_b0/best_model.pt`
- `projects/lung_colon/outputs/binary_resnet18/best_model.pt`

The Hugging Face token value was never read or displayed.

## Public Prediction Paths

### Leukemia

- real test image uploaded after BMP-to-PNG conversion outside the repository;
- predicted `leukemia_blast` with 71.51% confidence;
- probabilities displayed as 0.7151 and 0.2849, summing to 1.0000;
- ResNet18 loaded on demand and Grad-CAM rendered successfully.

### Breast

- real malignant BreakHis test image uploaded;
- predicted `benign` with 99.66% confidence;
- probabilities displayed as 0.9966 and 0.0034, summing to 1.0000;
- EfficientNet-B0 and Grad-CAM worked;
- patient-aware methodology, patient variability, magnification variability and the
  non-clinical limitations were visible.

This intentionally preserved high-confidence error is a model limitation, not a
deployment failure.

### Metastasis

- real metastatic PCam test image uploaded;
- predicted `metastatic` with 99.67% confidence;
- the two-class probability output and Grad-CAM rendered successfully;
- EfficientNet-B0 remained loaded while thresholds 0.30, 0.50 and 0.70 were tested;
- the original argmax and 99.67% metastatic probability stayed unchanged;
- only the separate threshold decision changed, without a new model load;
- the FP/FN comparison table and the explicit no-medical-threshold warning were visible.

### LungColon Multiclass

- explicit five-class mode selected;
- real `lung_adenocarcinoma` test image uploaded;
- predicted `lung_adenocarcinoma` with 99.96% confidence;
- five probabilities were displayed and summed to approximately 1 after rounding;
- EfficientNet-B0, the explicit mode label and Grad-CAM worked.

### LungColon Binary

- switching from multiclass unloaded the active model and removed the old prediction,
  probabilities and Grad-CAM;
- real malignant test image uploaded;
- predicted `malignant` with 100.00% displayed confidence;
- two probabilities were displayed and summed to approximately 1 after rounding;
- the distinct ResNet18 binary checkpoint and Grad-CAM worked.

Returning to Leukemia unloaded LungColon, cleared the previous output and left no model
in memory until an explicit load action.

## Controlled States

- prediction without an image is prevented by a disabled action button;
- a Markdown file was refused with `text/markdown files are not allowed` and no
  exception;
- changing project before prediction remained stable;
- changing LungColon mode before prediction cleared state and remained stable;
- the Metastasis threshold control appears only after a prediction, so it cannot be
  applied to a missing result.

## Cold Start And Layout

A Streamlit dashboard reboot was triggered after functional validation. The anonymous
public application became ready again in approximately 97 seconds. During this reboot,
the visible logs were dominated by dependency installation, including the CPU PyTorch
and OpenCV wheels. They did not provide enough evidence to claim that checkpoint
downloads were the main cold-start cost.

Desktop layout, sidebar controls, probability tables, threshold table and Grad-CAM
outputs rendered without a major horizontal overflow. The available browser controller
did not provide an independent mobile viewport emulation, so phone-specific layout
remains a manual validation item rather than a claimed pass.

## Frozen References

- ML tag: `multicancer-v1`
- ML commit: `a51772da5d50eb8d758c01d737ea09f2f388cea6`
- deployment branch: `deploy/multicancer-v1-streamlit`
- private model repository: `amiralfefe/rccia-multicancer-models`
- required checkpoints: five, approximately 132.17 MiB total
- validated deployment Python: `3.11`

No ML pipeline, model architecture, preprocessing, metric, checkpoint or tag was
changed during this validation. This remains an educational portfolio demonstration
only, not a medical or diagnostic tool and not a clinically validated system.
