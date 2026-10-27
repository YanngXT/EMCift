# Implementation Notes

## Released Scope

Only the original full EMCift model is exposed. `models/emcift.py` retains the original `Learner` and `MultiPrototypeManager` implementations. `models/base.py` retains the original evaluation and serialization helpers. `my_utils/inc_net.py` contains the exact network classes used by the model, with the backbone factory restricted to the original ImageNet-21K ViT and adapter branch. The main-table datasets and supplemental PACS/iWildCam streams share this implementation.

The representation concatenates the frozen pretrained branch with the adapter-tuned branch. The trainable adapters, MLP training head, multi-center prototype inference, conditional domain-adversarial loss, class balancing, semantic metric objective, prototype drift compensation and herding/replay logic are not replaced by a simplified implementation.

## Memory

The main configurations retain one exemplar per observed class-domain pair. The original `memory_size` field is not a new fixed global cap: the actual replay selection uses `memory_per_class` for each observed view. Exemplar indices refer back to original training images, not to cached feature vectors. Feature statistics used to maintain prototypes are distinct from exemplar image replay.

## Metrics

Let the trainer's cumulative semantic top1 curve be `a[0], ..., a[T-1]`. Average incremental accuracy is `mean(a)`, and final accuracy is `a[-1]`. The trainer evaluates each phase's held-out data after every learned phase to construct the semantic accuracy matrix. Final forgetting averages the drop from historical best accuracy to final accuracy across old phases, excluding the newly introduced final phase.

The release reader uses these native fields. It does not use the final fragment's accuracy, task-macro accuracy as a substitute for cumulative accuracy, or the average of the sequence of intermediate forgetting values.

## Packaging Changes

- Remove absolute server import paths and the forced Hugging Face mirror.
- Expose only EMCift in the model factory and only the three main datasets in the entry point.
- Read the original stream YAML through an explicit portable path.
- Make dataset downloading/preprocessing opt-in and accept already prepared directories without repeated archive downloads.
- Reset log handlers between multiple seeds; isolate runtime outputs from tracked source files.
- Disable optional final visualization and large final checkpoint serialization in the supplied configs; either can be explicitly enabled. Both occur after model evaluation and do not alter the learning rule.
- Retain only the three selected original main-dataset configurations; keep the iWildCam/PACS configurations separately under `experiments/supplemental/`.
- Use the original PACS split and iWildCam loader, making only file roots and manifest paths portable. The iWildCam image selection and ordering are bundled verbatim, without images.

`SOURCE_MANIFEST.json` records source-relative file names and hashes. Source-to-release AST equivalence checks are performed outside the public repository during packaging. They are not a claim that full benchmark retraining was performed for the release candidate.
