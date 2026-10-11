#!/usr/bin/env bash
# Build the thesis PDF: copy the source files listed in the appendix, then run XeLaTeX via tectonic.
# Usage: bash thesis/build.sh   (from the repo root or from thesis/)
set -euo pipefail
cd "$(dirname "$0")"
ROOT=..
mkdir -p appendix/src

for f in fgtn_neck.py heads.py make_class_weight.py \
         ava22_swin-tiny_fgtn_32x2x1-10e.py ava22_swin-tiny_fgtn_subset.py \
         ava22_swin-tiny_fgtn-ms-acrn-focal_32x2x1-10e.py; do
  cp "$ROOT/projects/fgtn_ava/$f" appendix/src/
done

# log excerpt: environment header + the ten per-epoch validation summaries
{
  sed -n '1,12p' "$ROOT/trainning.log"
  sed -n '30,44p' "$ROOT/trainning.log"
  echo "..."
  grep -E "Epoch\(val\) +\[[0-9]+\]\[4846/4846\]" "$ROOT/trainning.log"
} > appendix/src/log_excerpt.txt

if command -v tectonic >/dev/null; then
  tectonic --keep-logs main.tex
else
  latexmk -xelatex -interaction=nonstopmode main.tex
fi
echo "-> thesis/main.pdf"
