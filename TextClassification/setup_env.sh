#!/usr/bin/env bash
set -euo pipefail

# Creates a fresh venv (using the newest available Python on PATH),
# installs Jupyter deps + ML stack + CUDA 12.4 PyTorch wheels.

# --------- pick newest python available ----------
pick_python() {
  local candidates=(
    python3.13 python3.12 python3.11 python3.10 python3.9 python3
  )
  for c in "${candidates[@]}"; do
    if command -v "$c" >/dev/null 2>&1; then
      echo "$c"
      return 0
    fi
  done
  echo "No python3 found on PATH" >&2
  return 1
}

PY_BIN="$(pick_python)"

# --------- create venv ----------
if [ ! -d ".venv" ]; then
  "$PY_BIN" -m venv .venv
fi

# --------- activate venv ----------
# shellcheck disable=SC1091
source .venv/bin/activate

# --------- upgrade packaging tools ----------
python -m pip install -U pip setuptools wheel

# --------- jupyter basics ----------
pip install notebook ipykernel

# --------- core deps ----------
pip install \
  numpy \
  pandas \
  matplotlib \
  seaborn \
  scikit-learn \
  datasets \
  transformers \
  evaluate

# --------- accelerate ----------
pip install 'accelerate>=0.26.0'

# --------- pytorch (CUDA 12.4 wheels) ----------
pip install -U "torch>=2.6" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# --------- misc ----------
pip install -U protobuf
pip install -U sentencepiece

# --------- quick sanity check ----------
python - <<'PY'
import torch
print("Python:", __import__("sys").version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
