# Sourced by the launchers in scripts/; not meant to be run directly.

BENCHMARKS="edos edos_stable phdos"

# Move to the repository root. sbatch runs a copy of the launcher from Slurm's spool
# directory, so in that case use the submission directory instead.
cd_repo_root() {
    local here
    here="$(cd "$(dirname "$1")" && pwd)"
    if [[ -f "$here/common.sh" ]]; then
        cd "$here/.."
    else
        cd "${SLURM_SUBMIT_DIR:?}"
    fi
    if [[ ! -f train_stage1.py ]]; then
        echo "Cannot find the repository root; submit jobs from the repository root." >&2
        exit 1
    fi
}

# Benchmark name -> --data_src and --label_dim for the Python scripts.
set_benchmark() {
    case "$1" in
        edos)        DATA_SRC=binned_dos_128;             LABEL_DIM=128 ;;
        edos_stable) DATA_SRC=binned_dos_128_stable_only; LABEL_DIM=128 ;;
        phdos)       DATA_SRC=ph_dos_51;                  LABEL_DIM=51  ;;
        *)
            echo "Unknown benchmark '$1'. Expected one of: $BENCHMARKS" >&2
            exit 1 ;;
    esac
}

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
