# CrackFM

**A promptable foundation-model approach to crack segmentation and severity quantification.**

CrackFM adapts a segmentation foundation model (SAM 2) to thin, branching crack
structures using lightweight **LoRA / adapter** fine-tuning, and introduces
**geometry-adaptive prompting** — prompts (points/boxes) generated along the
medial axis of crack regions so that elongated, sub-pixel-thin defects are
segmented far more reliably than with naive box prompts. A downstream
**severity head** turns the predicted mask into physical quantities (crack
width, length, and a severity class) via skeleton + distance-transform analysis.

The repository is **runnable end-to-end without any gated weights**: a compact
U-Net baseline ships as a fallback so you can train and evaluate immediately on
free GPUs (Kaggle 2×T4), then swap in the SAM 2 backbone when you want the full
foundation-model pipeline.

---

## Why this exists

Crack inspection of civil infrastructure (pavement, bridges, masonry) is the
canonical "thin elongated structure" segmentation problem. General segmentation
models do poorly out of the box because:

1. cracks are only a few pixels wide, so a single bounding-box prompt covers
   mostly background;
2. severity (how wide / how long) — not just presence — is what engineers act
   on; and
3. labelled data is small and domain-shifted across datasets.

CrackFM targets all three: foundation-model priors for data efficiency,
geometry-adaptive prompting for thin structures, and an explicit severity head.

## Architecture

```
            image
              │
              ▼
   ┌──────────────────────┐
   │ coarse mask proposal  │   (edge/ridge heuristic or model logits)
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ geometry-adaptive     │   medial-axis point sampling + per-CC boxes,
   │ prompting             │   density ∝ local curvature / arc-length
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ SAM 2 (LoRA) | U-Net  │   promptable foundation backbone (or fallback)
   └──────────┬───────────┘
              ▼
        crack mask  ──►  severity head (width / length / class)
```

## Layout

```
src/crackfm/
  data/         dataset + transforms (image/mask folder pairs)
  models/       SAM 2 wrapper, LoRA, U-Net fallback, build()
  prompting/    geometry-adaptive prompt generation (the novelty)
  severity/     skeleton + distance-transform width/length/severity
  metrics.py    IoU / Dice / F1 + boundary-F (ODS/OIS style)
  losses.py     Dice + BCE + boundary/Tversky
  engine.py     plain-PyTorch trainer (no heavy framework dep)
  train.py      CLI entrypoint
  eval.py       CLI entrypoint
configs/        OmegaConf YAML (CLI-overridable)
scripts/        dataset download helpers
notebooks/      Kaggle entrypoint
tests/          pure-numpy tests for metrics/prompting/severity
docs/           research proposal & related work
```

## Quickstart

```bash
pip install -e ".[dev]"            # or: pip install -r requirements.txt
bash scripts/download_data.sh crack500 ./data
python -m crackfm.train  data.root=./data/crack500  model.name=unet  trainer.epochs=20
python -m crackfm.eval   ckpt=runs/last.pt  data.root=./data/crack500
pytest -q                          # pure-numpy tests, no GPU needed
```

To use the foundation backbone instead of the fallback:

```bash
pip install -e ".[sam2]"
python -m crackfm.train model.name=sam2_lora model.sam2_checkpoint=/path/to/sam2.pt
```

## Datasets

`scripts/download_data.sh` documents and (where licensing allows) fetches:
Crack500, DeepCrack, CrackForest (CFD), GAPs384. Each is normalised to a simple
`images/` + `masks/` folder pair with matching filenames and binary masks.

## Experiment tracking

Weights & Biases is wired in but **optional**: if `WANDB_API_KEY` is unset (or
`wandb` is not installed) the logger silently falls back to a no-op + local
CSV, so runs never block on an account.

## Status

Core scaffold + algorithms (prompting, severity, metrics, losses, trainer,
U-Net baseline) are implemented and unit-tested. SAM 2 LoRA wrapper is in place
behind an optional dependency. See `docs/proposal.md` for the research framing,
related work, and evaluation plan.
