#!/usr/bin/env bash
# Bumps `version` and `date-released` in CITATION.cff to <version> and
# today's date (UTC).
#
# Usage: bump-citation.sh <version>

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION="${1:?usage: $0 <version>}"
TODAY=$(date -u +%F)

sed -e "s/^version: .*/version: ${VERSION}/" \
    -e "s/^date-released: .*/date-released: '${TODAY}'/" \
    CITATION.cff > CITATION.cff.new
mv CITATION.cff.new CITATION.cff
grep -E "^(version|date-released):" CITATION.cff
