#!/usr/bin/env bash
# Regenerates environment_minimal.yaml and environment_dev.yaml from the
# conda deps declared in pyproject.toml's [tool.pixi.dependencies] /
# [tool.pixi.feature.dev.dependencies], via `pixi workspace export`.
#
# Some packages, currently genbadge and readme_renderer, have no conda-forge
# feedstock, so `--no-pypi` drops them entirely. To ensure that they are
# installed in new conda environments, they are appended to the environment
# yaml file in a `pip:` section.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

pixi workspace export conda-environment -e default --no-pypi -n oac_minimal environment_minimal.yaml
pixi workspace export conda-environment -e dev --no-pypi -n oac environment_dev.yaml

cat >> environment_dev.yaml <<'EOF'
- pip
- pip:
    # dependencies only available with pip
    - genbadge[coverage]
    - readme_renderer[md]
EOF
