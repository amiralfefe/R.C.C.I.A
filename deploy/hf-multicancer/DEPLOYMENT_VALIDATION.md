# MultiCancer D2 Deployment Validation

Validation date: 2026-08-07

## Status

**Partially completed, blocked before Space creation.**

The private model repository and its five required checkpoints were published and
verified. Creation of the public Docker Space was rejected by Hugging Face with HTTP
`402 Payment Required`: hosting Docker Spaces on free `cpu-basic` requires a PRO
subscription for this account.

No alternative SDK or architecture was created. The frozen ML release remains
`multicancer-v1` at commit `a51772da5d50eb8d758c01d737ea09f2f388cea6`.

## Hugging Face Resources

- Namespace: `amiralfefe`
- Model repository: `amiralfefe/rccia-multicancer-models`
- Model repository visibility: private
- Public Space repository: not created
- Public Space URL: unavailable
- `HF_MODEL_REPO_ID`: not configured because the Space does not exist
- `HF_TOKEN` Space secret: not configured

The available local authentication was used only for repository publication. It was not
reused as the Space secret because no dedicated read-only token was identified.

## Verified Checkpoints

| Remote path | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `leukemia/best_model.pt` | 44,788,299 | `344895dff67798aabdc6e6b57d4ad73e523ea0c34050c353fe429ff682fd36fb` |
| `breast/efficientnet_b0/best_model.pt` | 16,335,161 | `374797f7048494b39e8755b5a0c95c6dfa05cf406c33433ae339d16a50684401` |
| `metastasis/efficientnet_b0/best_model.pt` | 16,335,161 | `5b5f7ad0940bdff8b4e4888ed72fa7d7f51d97a2ec89f8f177a6f7fdfb520171` |
| `lung_colon/multiclass/efficientnet_b0/best_model.pt` | 16,350,649 | `37a4f290d669a521670b6fd07a8d43642cc60fe37ff34a71db272e371bdf842a` |
| `lung_colon/binary/resnet18/best_model.pt` | 44,788,299 | `84cf726e53c1dee3f758cfa8ea470dcf1339be644a5cd86424486c4258d5e789` |

The remote API exposed matching sizes and LFS SHA-256 values for all five files. Only
these checkpoints, the model card and the repository-managed `.gitattributes` file are
present in the private model repository.

## Validation Not Yet Run

The following checks require a running public Space and remain pending:

- Docker build and runtime status;
- cold start and checkpoint download timing;
- Leukemia, Breast, Metastasis and both LungColon paths;
- probabilities, Grad-CAM, threshold exploration and model unload behavior;
- degraded UI states;
- desktop and mobile browser validation.

## Required Resume Point

After the account can create Docker Spaces, resume from creation of the public
`amiralfefe/rccia-multicancer` Docker Space. Then configure `HF_MODEL_REPO_ID`, add a
dedicated read-only `HF_TOKEN` secret, publish the deployment bundle and run the pending
public validation.

This deployment remains an educational portfolio demonstration only. It is not a
medical device, diagnostic tool or clinically validated system.
