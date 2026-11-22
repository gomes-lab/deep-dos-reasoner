#!/bin/bash
# Fetch the prepared crystal graphs of this repository's benchmarks from Hugging Face.
#
#   bash scripts/download_data.sh              # eDOS and phDOS graphs (~200 MB compressed)
#   bash scripts/download_data.sh phdos        # selected graph sets: edos | phdos
#
# The benchmark definitions (manifest.csv, split.pkl) are tracked in git under
# data/<benchmark>/; this script only adds data/<set>/graphs/<id>.chkpt.
# The edos_stable benchmark reuses the edos graphs.
# Set HF_TOKEN to authenticate if needed; set HF_REPO (or HF_BASE_URL) to use a mirror.
set -euo pipefail
cd "$(dirname "$0")/.."

HF_REPO="${HF_REPO:-yingheng/DeepDOSReasoner}"
BASE_URL="${HF_BASE_URL:-https://huggingface.co/datasets/${HF_REPO}/resolve/main}"
GRAPH_SETS="edos phdos"

if [[ $# -gt 0 ]]; then sets=("$@"); else read -r -a sets <<< "$GRAPH_SETS"; fi
for s in "${sets[@]}"; do
    case "$s" in
        edos|phdos) ;;
        edos_stable) echo "edos_stable uses the edos graphs; downloading edos." >&2 ;;
        *) echo "Unknown graph set '$s'. Expected one of: $GRAPH_SETS" >&2; exit 1 ;;
    esac
done

if command -v sha256sum >/dev/null; then sha256() { sha256sum "$@"; }; else sha256() { shasum -a 256 "$@"; }; fi
fetch() {
    if [[ -n "${HF_TOKEN:-}" ]]; then
        curl -fL --retry 3 -H "Authorization: Bearer ${HF_TOKEN}" -o "$2" "$BASE_URL/$1"
    else
        curl -fL --retry 3 -o "$2" "$BASE_URL/$1"
    fi
}

fetch SHA256SUMS data/SHA256SUMS
for s in "${sets[@]}"; do
    [[ "$s" == edos_stable ]] && s=edos
    echo "==> $s"
    fetch "$s/graphs.tar.gz" "data/$s/graphs.tar.gz"
    (cd data && grep " $s/graphs.tar.gz\$" SHA256SUMS | sha256 -c -)
    rm -rf "data/$s/graphs"
    tar -xzf "data/$s/graphs.tar.gz" -C "data/$s"
    rm "data/$s/graphs.tar.gz"
    echo "$s: $(find "data/$s/graphs" -name '*.chkpt' | wc -l | tr -d ' ') graphs"
done
