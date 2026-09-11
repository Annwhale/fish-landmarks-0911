# Third-party software, models and external data

The project `LICENSE` applies only to project-authored source code. It does
not relicense dependencies, pretrained weights, trained checkpoints or
external datasets. The delivery archives intentionally exclude model weight
files and downloaded external datasets.

## Components used for technical validation

| Component | Recorded version or model | Upstream terms recorded for this run | Redistribution in this package |
|---|---|---|---|
| Ultralytics | Python package 8.4.48; generic YOLO pose checkpoint supplied from a local path | Package metadata reports AGPL-3.0; users must review the current upstream terms and the provenance of any checkpoint they obtain | Package code and all YOLO weights are excluded |
| PyTorch | torch 2.4.1+cu124 | BSD-3-Clause in installed package metadata | Dependency not vendored |
| torchvision | 0.19.1+cu124; ResNet-34 backbone | BSD in installed package metadata | Dependency and pretrained weights not vendored |
| DINOv2 | `dinov2_vitl14_reg`, frozen backbone | The inspected upstream model card and repository state Apache License 2.0 for the standard DINOv2 code and model weights | Upstream code and weights not vendored |
| OpenCV Python | 4.10.0.84 | Apache 2.0 in installed package metadata | Dependency not vendored |
| pandas | 2.3.3 | BSD 3-Clause in installed package metadata | Dependency not vendored |
| PyArrow | 20.0.0 | Apache Software License in installed package metadata | Dependency not vendored |
| LocateAnything | `nvidia/LocateAnything-3B`, model release 2026-05-26 | NVIDIA License for non-commercial academic and non-profit research; Qwen2.5-3B-Instruct and MoonViT components retain their upstream terms | Model code and weights are excluded; users obtain them from NVIDIA |
| Transformers | 4.57.3 for the LocateAnything run | Apache License 2.0 in upstream package metadata | Dependency not vendored |
| PEFT / Accelerate | PEFT 0.20.0; Accelerate 1.14.0 | Apache License 2.0 in upstream package metadata | Dependencies not vendored |
| decord / LMDB | decord 0.6.0; lmdb 1.7.5 | Upstream package terms apply | Dependencies not vendored |

The environment file is a reproducibility aid, not a substitute for the
licence notices distributed by each dependency. Users who reproduce the
experiments must obtain dependencies and pretrained weights from their
official sources and comply with the terms in effect for those artefacts.

## External validation data

Melops version 2.0 is an independently released Zenodo dataset:
`10.5281/zenodo.17404087`. Its images, annotations and metadata are not
redistributed here. Only project-authored crosswalk tables, audit summaries
and aggregate evaluation outputs are included. Users must obtain the exact
dataset version from Zenodo and follow its recorded terms and attribution
requirements.

Ravan, 3D-ZeF and other inspected resources are likewise not redistributed.
Their mention in audit documentation does not add them to the project data
licence.
