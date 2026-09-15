#!/usr/bin/env bash
# Bumps __version__ in openairclim/__about__.py. Shared by prepare-release.yml
# (real release) and publish.yml (for TestPyPI).
#
# Usage: bump-version.sh <version>

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION="${1:?usage: $0 <version>}"

sed "s/^__version__ = .*/__version__ = \"${VERSION}\"/" openairclim/__about__.py > openairclim/__about__.py.new
mv openairclim/__about__.py.new openairclim/__about__.py
grep __version__ openairclim/__about__.py
