#!/usr/bin/env bash
# Generates a categorised changelog entry for a release and prepends it to
# CHANGELOG.md. Release notes come from GitHub's "generate release notes"
# API, categorised using .github/release.yml.
#
# GitHub's raw output nests "## What's Changed" / "### <category>" at
# h2/h3. Since the entry is inserted under our own "## [version] - date"
# heading, every level is demoted by one (-> h3/h4) so that it nests correctly.
#
# Usage: generate-changelog-entry.sh <version> [previous_tag]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION="${1:?usage: $0 <version> [previous_tag]}"
PREVIOUS_TAG="${2:-}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set to owner/repo}"

NOTES=$(mktemp)
ENTRY=$(mktemp)
trap 'rm -f "$NOTES" "$ENTRY"' EXIT

gh api \
  --method POST \
  -H "Accept: application/vnd.github+json" \
  "repos/${GITHUB_REPOSITORY}/releases/generate-notes" \
  -f tag_name="v${VERSION}" \
  ${PREVIOUS_TAG:+-f previous_tag_name="$PREVIOUS_TAG"} \
  --jq .body \
  | grep -v '^<!-- Release notes generated using configuration' \
  | sed 's/^##/###/' \
  | awk '/./{f=1} f' \
  > "$NOTES"

TODAY=$(date -u +%F)
{
  echo "## [${VERSION}] - ${TODAY}"
  echo
  cat "$NOTES"
  echo
} > "$ENTRY"

awk -v entryfile="$ENTRY" 'NR==2{print; while((getline line < entryfile)>0) print line; next} 1' \
  CHANGELOG.md > CHANGELOG.md.new
mv CHANGELOG.md.new CHANGELOG.md
