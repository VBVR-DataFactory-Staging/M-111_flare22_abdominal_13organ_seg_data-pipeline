# M-111 scaffold TODO

Scaffolded from template: `M-037_amos_multi_organ_segmentation_data-pipeline` (2026-04-20)

## Status
- [x] config.py updated (domain=flare22_abdominal_13organ_seg, s3_prefix=M-111_FLARE22/raw/, fps=3)
- [ ] core/download.py: update URL / Kaggle slug / HF repo_id
- [ ] src/download/downloader.py: adapt to dataset file layout
- [ ] src/pipeline/_phase2/*.py: adapt raw → frames logic (inherited from M-037_amos_multi_organ_segmentation_data-pipeline, likely needs rework)
- [ ] examples/generate.py: verify end-to-end on 3 samples

## Task prompt
This abdominal CT slice (FLARE22). Segment 13 abdominal organs with distinct colors (liver, spleen, kidneys, pancreas, gallbladder, stomach, aorta, etc.).

Fleet runs likely FAIL on first attempt for dataset parsing; iterate based on fleet logs at s3://vbvr-final-data/_logs/.
