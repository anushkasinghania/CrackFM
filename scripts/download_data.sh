#!/usr/bin/env bash
# Download + normalise public crack-segmentation datasets into CrackFM's layout:
#
#   <out>/<dataset>/images/   and   <out>/<dataset>/masks/
#
# Usage:
#   bash scripts/download_data.sh <dataset> <out_dir>
#   bash scripts/download_data.sh crack500 ./data
#
# Datasets differ in licensing and hosting; some require manual download (a
# Kaggle account or an institutional mirror). Where automatic download is not
# permitted we print the canonical source so you can fetch it yourself, then
# re-run with --normalise to convert an extracted folder into the layout above.
set -euo pipefail

DATASET="${1:-}"
OUT="${2:-./data}"

usage() {
  cat <<EOF
Supported datasets:
  crack500    Pavement cracks (~500 imgs). Source: github.com/fyangneil/pavement-crack-detection
  deepcrack   DeepCrack benchmark.        Source: github.com/yhlleo/DeepCrack
  cfd         CrackForest Dataset.        Source: github.com/cuilimeng/CrackForest-dataset
  gaps384     GAPs384 subset.             Source: see GAPs dataset (TU Ilmenau), requires request

Each produces:  <out>/<dataset>/images/  and  <out>/<dataset>/masks/
EOF
}

if [[ -z "$DATASET" ]]; then usage; exit 1; fi
mkdir -p "$OUT"
DEST="$OUT/$DATASET"
mkdir -p "$DEST/images" "$DEST/masks"

case "$DATASET" in
  deepcrack)
    echo "[download] DeepCrack -> $DEST"
    TMP="$(mktemp -d)"
    git clone --depth 1 https://github.com/yhlleo/DeepCrack.git "$TMP/DeepCrack"
    # DeepCrack ships train/test image + label folders; flatten into images/ + masks/.
    find "$TMP/DeepCrack" -path '*train_img*' -type f \( -name '*.jpg' -o -name '*.png' \) \
      -exec cp {} "$DEST/images/" \; || true
    find "$TMP/DeepCrack" -path '*train_lab*' -type f -name '*.png' \
      -exec cp {} "$DEST/masks/" \; || true
    rm -rf "$TMP"
    ;;
  cfd)
    echo "[download] CrackForest (CFD) -> $DEST"
    TMP="$(mktemp -d)"
    git clone --depth 1 https://github.com/cuilimeng/CrackForest-dataset.git "$TMP/CFD"
    cp "$TMP/CFD/image/"*.jpg "$DEST/images/" 2>/dev/null || true
    echo "[note] CFD ground truth is in MATLAB .mat (seg) format under $TMP/CFD/groundTruth;"
    echo "       convert to binary PNG masks (same stem) into $DEST/masks/ before training."
    echo "       (kept the clone at: $TMP/CFD)"
    ;;
  crack500)
    echo "[crack500] Automatic download is not redistributable here."
    echo "  1) Get it from: https://github.com/fyangneil/pavement-crack-detection (CRACK500)"
    echo "     or the Kaggle mirror 'crack500'."
    echo "  2) Put images in:  $DEST/images/   and binary masks in:  $DEST/masks/"
    echo "     (matching filenames by stem; masks white=crack, black=background)."
    ;;
  gaps384)
    echo "[gaps384] GAPs requires a usage agreement (TU Ilmenau)."
    echo "  Request access, then place images/masks into $DEST/images and $DEST/masks."
    ;;
  *)
    usage; exit 1 ;;
esac

echo "[done] $DATASET prepared under $DEST"
echo "       images: $(find "$DEST/images" -type f | wc -l)  masks: $(find "$DEST/masks" -type f | wc -l)"
