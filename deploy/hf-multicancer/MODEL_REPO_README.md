---
library_name: pytorch
tags:
  - image-classification
  - computer-vision
  - educational
---

# R.C.C.I.A MultiCancer Checkpoints

This private repository stores the five checkpoints required by the public R.C.C.I.A
MultiCancer portfolio demo.

The files come from four distinct specialized image-classification pipelines:

- Leukemia;
- Breast;
- Metastasis;
- LungColon multiclass;
- LungColon binary.

They do not form a universal multicancer model. The public hub explicitly routes each
image to a user-selected specialized pipeline with its own classes, preprocessing,
checkpoint and methodological limits.

The datasets are not included. The source code and project documentation are maintained
in the [R.C.C.I.A GitHub repository](https://github.com/amiralfefe/R.C.C.I.A).

## Intended Use

These checkpoints support an educational AI/data portfolio demonstration only. They are
not clinically validated and must not be used as a medical device, diagnostic tool,
decision aid or source of health advice.
