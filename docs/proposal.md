# CrackFM: Geometry-Adaptive Foundation-Model Prompting for Crack Segmentation and Severity Quantification

**Research proposal / related-work brief**
Author: Anushka Singhania

---

## Abstract

Surface-crack inspection of civil infrastructure — pavement, bridges, tunnels,
masonry is a high-value computer-vision problem that remains hard for three
structural reasons: cracks are *thin* (often 1–3 px wide), *elongated and
branching*, and *severely class-imbalanced* against textured backgrounds; labels
are *scarce and domain-shifted* across datasets; and practitioners need
*severity* (width, length, extent), not merely a binary "crack/no-crack" map.
Promptable segmentation foundation models such as SAM 2 offer strong priors and
data efficiency, but their default prompting (a box or a single click) is a poor
fit for thin structures. **CrackFM** proposes (i) a *geometry-adaptive prompting*
module that derives point/box prompts from the medial axis and curvature of a
coarse crack proposal, (ii) parameter-efficient **LoRA** adaptation of a frozen
SAM 2 backbone for small crack datasets, and (iii) an explicit **severity head**
that converts predicted masks into physical crack width/length/severity via
skeleton + distance-transform analysis. We target measurable gains over U-Net /
DeepCrack-style baselines on IoU/Dice and, critically, on a *boundary-F* (ODS/OIS)
protocol appropriate for thin structures, plus a quantitative severity
evaluation that prior segmentation-only work does not report.

---

## 1. Motivation

Manual crack inspection is slow, subjective and unsafe; automated visual
inspection is therefore an active research and industrial area. Yet
general-purpose segmentation models underperform on cracks because the object of
interest violates the assumptions those models are tuned for:

1. **Thinness.** A crack a few pixels wide means any axis-aligned bounding box is
   dominated by background; a single foreground click captures a vanishing
   fraction of the structure. Promptable models (SAM/SAM 2) are thus handicapped
   by their *prompt interface*, not only their weights.
2. **Severity, not presence.** Engineering decisions hinge on *how wide* and *how
   long* a crack is. A binary mask alone is not actionable; width/length must be
   recovered in physical units.
3. **Data scarcity and domain shift.** Public crack datasets are small (hundreds
   to low thousands of images) and visually heterogeneous (pavement vs. concrete
   vs. masonry, different cameras, lighting). Full fine-tuning of large models
   overfits; data-efficient adaptation is required.

CrackFM is designed around exactly these three pressure points.

## 2. Related Work

**Classical and CNN crack segmentation.** CrackForest (CFD) framed road-crack
detection with structured random forests over hand-crafted features [Shi 2016].
The deep-learning wave produced dedicated architectures: DeepCrack learns
hierarchical convolutional features with deep supervision for pixel-wise crack
detection [Zou 2019; Liu 2019], and FPHBN introduced feature-pyramid +
hierarchical boosting for pavement cracks, releasing the widely used **CRACK500**
benchmark [Yang 2019]. U-Net [Ronneberger 2015] and its encoder–decoder
descendants remain strong, common baselines. These methods establish the metrics
and datasets but (a) train from scratch or from ImageNet, limiting data
efficiency, and (b) stop at the binary mask.

**Segmentation foundation models.** SAM [Kirillov 2023] introduced
promptable segmentation at scale; SAM 2 [Ravi 2024] extends it to images and
video with a stronger Hiera image encoder and a streaming memory design.
Self-supervised encoders such as DINOv2 [Oquab 2023] provide transferable dense
features. The promise for cracks is data efficiency via strong priors; the gap is
that out-of-the-box prompting (box / single point) and zero-shot weights are
ill-matched to thin, branching structures, and recent applications of SAM to
cracks largely rely on naive prompts or full fine-tuning.

**Parameter-efficient adaptation.** LoRA [Hu 2022] adapts large frozen models by
training low-rank updates, drastically reducing trainable parameters and
overfitting on small datasets directly relevant to the crack data regime.

**Thin-structure evaluation.** Exact pixel overlap is too strict for structures a
few pixels wide; the edge/contour literature uses tolerance-based matching and
the ODS/OIS protocol [Arbeláez 2011], which we adopt as a primary metric.

**Severity quantification.** Crack *width* is commonly estimated from the
distance transform sampled along the skeleton/medial axis of a mask; *length*
from skeleton arc-length. This is standard in measurement pipelines but is
rarely reported jointly with a learned segmentation model under a single,
reproducible protocol.

## 3. Research Gap and Contributions

The literature optimises binary crack masks (often from scratch) and, separately,
measures width from masks. What is missing is a **single foundation-model
pipeline that (a) is prompted in a way suited to thin structures, (b) is adapted
data-efficiently, and (c) reports severity as a first-class, evaluated output.**

CrackFM's contributions:

1. **Geometry-adaptive prompting.** Prompts are generated from the *geometry* of
   a coarse crack proposal: positive points sampled along the **medial axis**
   with density proportional to local **curvature/branching**, tight **per
   connected-component boxes**, and **background-margin negatives**. This matches
   prompt coverage to the crack itself rather than to a bounding box.
   (Implemented and unit-tested: `src/crackfm/prompting/`.)
2. **LoRA-adapted SAM 2 backbone** with a frozen image encoder, for
   data-efficient transfer to small, heterogeneous crack datasets.
   (`src/crackfm/models/`.)
3. **Severity head** producing width (mean/max/p95), length and a thresholded
   severity class in physical units when a mm/px scale is supplied.
   (Implemented and unit-tested: `src/crackfm/severity/`.)
4. **A thin-structure evaluation protocol** combining region metrics
   (IoU/Dice/F1) with a tolerance-based **boundary-F** (ODS/OIS-style) and a
   severity report — all reproducible from config.

## 4. Method

**Pipeline.** Given an image, a coarse crack proposal (model logits, or a
ridge/edge heuristic at inference time) is converted by the geometry-adaptive
prompter into a `PromptSet` (points + boxes). SAM 2 (image encoder frozen +
LoRA; mask decoder fine-tuned) consumes the image and prompts to produce crack
logits. A combined **Dice + BCE + boundary + Tversky** loss (recall-biased for
the rare foreground; boundary-weighted near GT edges) supervises training. At
evaluation, the predicted mask is passed to the severity head.

**Geometry-adaptive prompting (core).** From a binary proposal we compute the
skeleton (medial axis); curvature is approximated by deviation of each skeleton
pixel's neighbour count from 2, concentrating prompts at bends, tips and
branch points. Negatives are drawn from a dilation ring just outside the crack to
suppress leakage into shadows/texture. Output maps directly onto SAM 2's
`point_coords` / `point_labels` / `box` API.

**Severity.** The Euclidean distance transform sampled on the skeleton gives the
local half-width; doubling yields width along the centreline (mean/max/p95).
Skeleton length approximates crack length; with `mm_per_pixel`, outputs are in
millimetres and mapped to a low/medium/high severity class by width thresholds.

**Baseline / fallback.** A compact **U-Net** ships so the full pipeline (data →
train → eval → severity) runs end-to-end on free GPUs with no gated weights,
giving both an honest baseline and a smoke-test of the whole stack.

## 5. Datasets

| Dataset       | Domain            | Notes                                   |
|---------------|-------------------|-----------------------------------------|
| CRACK500      | Pavement          | FPHBN benchmark, pixel masks            |
| DeepCrack     | Concrete/pavement | Deep-supervision benchmark              |
| CrackForest   | Road              | CFD, classical benchmark                |
| GAPs384       | Pavement          | Subset of GAPs (access agreement)       |

All are normalised by `scripts/download_data.sh` into a uniform
`images/` + `masks/` layout. Cross-dataset (train-on-A, test-on-B) experiments
probe domain robustness — a setting where foundation-model priors should help.

## 6. Evaluation Protocol

* **Region:** IoU, Dice, precision/recall/F1.
* **Thin-structure:** boundary-F at pixel tolerances {0,1,2}, reported in the
  ODS/OIS spirit (dataset-optimal vs. image-optimal thresholds).
* **Severity:** mean absolute error of estimated max-width vs. annotated width
  (where available) and severity-class accuracy.
* **Data efficiency:** metric vs. training-set fraction (10/25/50/100%), to test
  the central claim that foundation-model + LoRA wins in the low-data regime.

**Baselines:** U-Net (ours), DeepCrack-style supervised CNN, SAM 2 zero-shot with
box prompts, SAM 2 zero-shot with naive point prompts. **Ablations:** prompting
strategy (none / uniform medial-axis / curvature-adaptive), with vs. without
per-component boxes, with vs. without negatives, LoRA rank, and loss-term
contributions.

## 7. Expected Outcomes

We expect geometry-adaptive prompting to substantially beat box/naive-point SAM 2
on thin cracks (largest gains on boundary-F and recall), LoRA adaptation to match
or exceed full fine-tuning while training a small fraction of parameters
(especially at low data fractions), and the severity head to deliver usable
width/length estimates that segmentation-only baselines do not provide.

## 8. Work Plan

1. **Data + baseline (done/in progress):** dataset normalisation, U-Net baseline,
   metric/severity/prompting modules with unit tests. *(Implemented.)*
2. **SAM 2 integration:** wire image-encoder (LoRA) + mask decoder; feed
   geometry-adaptive prompts; reproduce zero-shot baselines.
3. **Experiments:** full benchmark + ablations + data-efficiency curves on
   Kaggle (2×T4); optional larger runs on cloud GPU.
4. **Writing:** convert this brief into the paper's Introduction + Related Work +
   Method; add results, figures, ablation tables.

## 9. Target Venues

Workshops/venues on infrastructure inspection and applied CV (e.g.
CVPR/ICCV/WACV workshops on vision for construction/transportation; journals such
as *Automation in Construction*, *Computer-Aided Civil and Infrastructure
Engineering*).

---

## References

- Arbeláez, Maire, Fowlkes, Malik. *Contour Detection and Hierarchical Image
  Segmentation.* IEEE TPAMI, 2011.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
  arXiv:2106.09685.
- Kirillov et al. *Segment Anything.* ICCV 2023. arXiv:2304.02643.
- Liu et al. *DeepCrack: A deep hierarchical feature learning architecture for
  crack segmentation.* Neurocomputing, 2019.
- Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.*
  2023. arXiv:2304.07193.
- Ravi et al. *SAM 2: Segment Anything in Images and Videos.* 2024.
  arXiv:2408.00714.
- Ronneberger, Fischer, Brox. *U-Net: Convolutional Networks for Biomedical Image
  Segmentation.* MICCAI 2015. arXiv:1505.04597.
- Shi et al. *Automatic Road Crack Detection Using Random Structured Forests
  (CrackForest).* IEEE T-ITS, 2016.
- Yang et al. *Feature Pyramid and Hierarchical Boosting Network for Pavement
  Crack Detection (FPHBN / CRACK500).* IEEE T-ITS, 2019. arXiv:1901.06340.
- Zou et al. *DeepCrack: Learning Hierarchical Convolutional Features for Crack
  Detection.* IEEE TIP, 2019.

*References are provided for orientation; verify exact bibliographic details and
add recent SAM-for-cracks works at write-up time.*
