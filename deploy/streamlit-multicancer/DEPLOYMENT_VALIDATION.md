# Streamlit MultiCancer Deployment Validation

Validation date: 2026-08-07

## Pre-Publication Status

| Check | Status |
| --- | --- |
| Packaging files | pass |
| Dedicated entrypoint | pass, local HTTP 200 |
| Local checkpoints | 5/5 present |
| Downloader mock tests | 6 passed |
| Monorepo tests | 120 passed |
| Sensitive/heavy file scan | pass |
| Public URL | pending |
| Cold start | pending |
| Public five-path validation | pending |
| Mobile validation | pending |

The public application has not been created yet. No URL, cold-start duration or public
behavior is claimed in this report.

The local Streamlit server returned HTTP `200`. Streamlit AppTest executed the dedicated
entrypoint without exception and navigated through Leukemia, Breast, Metastasis,
LungColon multiclass and LungColon binary. No downloader output appeared because all
five local checkpoints were already present. The existing real-checkpoint QA tests also
remained green.

## Frozen References

- ML tag: `multicancer-v1`
- ML commit: `a51772da5d50eb8d758c01d737ea09f2f388cea6`
- Private model repository: `amiralfefe/rccia-multicancer-models`
- Required checkpoints: five, approximately 132.17 MiB total
- Validated local Python: `3.11.8`

The deployment bootstrap does not load models or alter classes, preprocessing,
architectures, metrics, checkpoints or threshold behavior.

## Pending Public Validation

- first opening and checkpoint download behavior;
- Leukemia prediction, probabilities and Grad-CAM;
- Breast prediction, probabilities, Grad-CAM and methodology context;
- Metastasis prediction, Grad-CAM and threshold exploration without reinference;
- LungColon multiclass prediction, five probabilities and Grad-CAM;
- LungColon binary mode switch, two probabilities and Grad-CAM;
- return to Leukemia with correct unload behavior;
- invalid image and no-image UI states;
- desktop and mobile layout;
- resource-limit observations.

This remains an educational portfolio demonstration only, not a medical or diagnostic
tool and not a clinically validated system.
