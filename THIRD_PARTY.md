# Third-Party Notices

This repository contains EMCift and the shared components it requires, not implementations of the comparison methods. Original source references and copyright headers are retained. The following upstream projects are acknowledged for the referenced scaffolding and vision/data utilities:

| Component/reference | Upstream | Accompanying license text |
|---|---|---|
| Incremental-network/classifier utility scaffolding | https://github.com/G-U-N/PyCIL | `third_party/licenses/PyCIL-LICENSE.txt` |
| Parameter-efficient adapter reference | https://github.com/jxhe/unify-parameter-efficient-tuning | `third_party/licenses/unify-PET-LICENSE.txt` |
| ViT reference and runtime dependency | https://github.com/huggingface/pytorch-image-models | `third_party/licenses/timm-LICENSE.txt` |
| DeiT reference named in the original adapter source | https://github.com/facebookresearch/deit | `third_party/licenses/DeiT-LICENSE.txt` |
| MAE/global-pooling reference named in the original adapter source | https://github.com/facebookresearch/mae | `third_party/licenses/MAE-LICENSE.txt` |
| Dataset/IDX reader utilities | https://github.com/pytorch/vision | `third_party/licenses/torchvision-LICENSE.txt` |

The included license texts are upstream notices, not a blanket relicensing of EMCift. In particular, the MAE reference has non-commercial terms; do not assume that every referenced component is MIT or Apache licensed. Dataset and pretrained-weight permissions are separate from code permissions.

The main-table packaging removes unrelated methods, server paths and auxiliary constructors; relevant source-file modification notes and upstream references are retained. Authors must confirm the license and distribution rights for their project-specific contributions before publishing this release candidate. No copyright ownership or permission is invented by the packaging process.
