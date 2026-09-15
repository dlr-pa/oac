#!/usr/bin/env bash
# Regenerates environment_minimal.yaml and environment_dev.yaml from the
# conda deps declared in pyproject.toml's [tool.pixi.dependencies] /
# [tool.pixi.feature.dev.dependencies], via `pixi workspace export`.
#
# Some packages are installed through pip. Here we provide a bit of background
# on why:
# - genbadge: does not have a conda-forge feedstock (yet)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

pixi workspace export conda-environment -e default --no-pypi -n oac_minimal environment_minimal.yaml
pixi workspace export conda-environment -e dev --no-pypi -n oac environment_dev.yaml

cat >> environment_dev.yaml <<'EOF'
- pip
- pip:
    # dependencies only available with pip
    - genbadge[coverage]
EOF
