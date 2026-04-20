"""M-111: FLARE22 abdominal 13-organ segmentation.

Dataset: FLARE 2022 challenge training set (https://zenodo.org/records/7860267).
Raw layout (after extract to _extracted/M-111_FLARE22/):
    FLARE22Train/
        images/FLARE22_Tr_XXXX_0000.nii.gz
        labels/FLARE22_Tr_XXXX.nii.gz

Labels (13 organs):
    1=liver, 2=right kidney, 3=spleen, 4=pancreas, 5=aorta, 6=IVC,
    7=right adrenal, 8=left adrenal, 9=gallbladder, 10=esophagus,
    11=stomach, 12=duodenum, 13=left kidney.

Case B: abdominal CT axial slice sequence, FPS=3 (per HARNESS fps=3 for 3D volumes).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import nibabel as nib
from common import (
    DATA_ROOT, window_ct, to_rgb, overlay_multi, write_task,
    COLORS, fit_square, pick_annotated_idx,
)

PID = "M-111"
TASK_NAME = "flare22_abdominal_13organ_seg"
FPS = 3

ORGANS = [
    ("liver",               "green"),
    ("right_kidney",        "orange"),
    ("spleen",              "purple"),
    ("pancreas",            "pink"),
    ("aorta",               "red"),
    ("IVC",                 "blue"),
    ("right_adrenal_gland", "teal"),
    ("left_adrenal_gland",  "teal"),
    ("gallbladder",         "lime"),
    ("esophagus",           "cyan"),
    ("stomach",             "yellow"),
    ("duodenum",            "brown"),
    ("left_kidney",         "orange"),
]
COLOR_LIST = [(n, COLORS[c]) for n, c in ORGANS]

PROMPT = (
    "This is an abdominal CT scan from the FLARE 2022 dataset. "
    "Segment all 13 abdominal organs simultaneously: liver (green), "
    "right/left kidneys (orange), spleen (purple), pancreas (pink), "
    "aorta (red), IVC (blue), adrenal glands (teal), gallbladder (lime), "
    "esophagus (cyan), stomach (yellow), duodenum (brown). "
    "Overlay each organ with its assigned color and draw contour boundaries on every slice."
)


def process_case(img_path: Path, lbl_path: Path, task_idx: int):
    img_vol = np.transpose(nib.load(str(img_path)).get_fdata(), (2, 1, 0))
    lbl_vol = np.transpose(nib.load(str(lbl_path)).get_fdata(), (2, 1, 0)).astype(np.int32)

    n = img_vol.shape[0]
    step = max(1, n // 60)
    indices = list(range(0, n, step))[:60]

    first_frames, last_frames, gt_frames, flags = [], [], [], []
    for z in indices:
        ct = window_ct(img_vol[z])
        rgb = to_rgb(ct)
        rgb = fit_square(rgb, 512)
        lab = lbl_vol[z].astype(np.int32)
        lab_square = fit_square(lab.astype(np.int16), 512).astype(np.int32)
        ann = overlay_multi(rgb, lab_square, COLOR_LIST)
        first_frames.append(rgb)
        last_frames.append(ann)
        has = bool((lab_square > 0).any())
        flags.append(has)
        if has:
            gt_frames.append(ann)
    if not gt_frames:
        gt_frames = last_frames[:5]
    pick = pick_annotated_idx(flags)
    first_frame = first_frames[pick]
    final_frame = last_frames[pick]

    meta = {
        "task": "FLARE22 abdominal 13-organ segmentation",
        "dataset": "FLARE 2022",
        "case_id": img_path.stem.replace("_0000", "").replace(".nii", ""),
        "modality": "CT",
        "organs": [n for n, _ in ORGANS],
        "colors": {n: c for n, c in ORGANS},
        "fps_source": "derived (case B slice sequence, fps=3 per HARNESS)",
        "num_slices_total": int(n),
        "num_slices_used": len(indices),
        "source_split": "train",
    }
    return write_task(PID, TASK_NAME, task_idx,
                      first_frame, final_frame,
                      first_frames, last_frames, gt_frames,
                      PROMPT, meta, FPS)


def main():
    root = DATA_ROOT / "_extracted" / "M-111_FLARE22" / "FLARE22Train"
    cases = sorted(root.glob("images/FLARE22_Tr_*_0000.nii.gz"))
    print(f"  {len(cases)} FLARE22 cases")
    for i, img in enumerate(cases):
        # image FLARE22_Tr_0001_0000.nii.gz → label FLARE22_Tr_0001.nii.gz
        label_name = img.name.replace("_0000.nii.gz", ".nii.gz")
        lbl = root / "labels" / label_name
        if not lbl.exists():
            print(f"  skip {img.name}: no label")
            continue
        d = process_case(img, lbl, i)
        print(f"  wrote {d}")


if __name__ == "__main__":
    main()
