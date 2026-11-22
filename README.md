# DeepDOSReasoner

Code for **DeepDOSReasoner: Physics-grounded reasoning for density-of-states prediction and materials discovery**.

Yingheng Wang, Francesco Ricci, Zhilong Wang, Tao Yu, Junwen Bai, Xiangyun Lei, Shufeng Kong, Fengqi You, John M. Gregoire, Carla P. Gomes

[[Project page]](https://gomes-lab.github.io/deepdosreasoner-demo/) [[Benchmark graphs]](https://huggingface.co/datasets/yingheng/DeepDOSReasoner) [[Predicted Materials Project spectra]](https://huggingface.co/datasets/yingheng/mp-predicted-edos-phdos)

DeepDOSReasoner maps a crystal structure to its electronic or phonon density of states (eDOS, phDOS).

1. **Stage I, structure-aware spectrum decoding.** An edge-aware crystal-graph Transformer embeds the atoms and their neighbourhoods. A Transformer decoder then predicts the spectrum one energy bin at a time; each bin attends to the atom embeddings (atom-to-energy source attention).
2. **Stage II, constraint-preserving physical reasoning (CPPR).** With Stage I frozen, the normalized spectrum is first reweighted by a learned residual and then transported for K = 5 steps under a continuity equation, discretized with a CDF-based upwind scheme and no-flux boundaries. The learned velocity is clamped so that the Courant number never exceeds 1/2, which keeps each update non-negative; the scheme conserves the total number of states exactly.

The reported model averages two independently seeded Stage I/II pairs.

## Contents

| Path | Purpose |
| --- | --- |
| `train_stage1.py` | Train Stage I |
| `train_stage2.py` | Train Stage II on a frozen Stage I checkpoint |
| `test.py` | Score the two-seed ensemble on the test split |
| `parser.py` | Command-line options shared by the three entry points |
| `DOS_Reasoner/data.py` | Benchmark lookup and crystal-graph dataset |
| `DOS_Reasoner/stage1.py` | Graph encoder, spectrum decoder and Stage I loss |
| `DOS_Reasoner/stage2.py` | Residual and velocity networks, CPPR transport |
| `DOS_Reasoner/transformer.py` | Attention and decoder layers, Stage I learning-rate schedule |
| `DOS_Reasoner/utils.py` | R², MAE, MSE and Wasserstein distance |
| `data/` | Benchmark ID lists and fixed splits (graphs are downloaded into it) |
| `scripts/` | Data download, environment check, and bash/Slurm launchers |
| `environment.yml` | Conda environment |

## Setup

Requirements: Linux x86_64 and one NVIDIA GPU (training and evaluation do not run on CPU). The tested stack is Python 3.12, PyTorch 2.5.1 with CUDA 12.4, and PyTorch Geometric 2.6.1, which supports compute capability sm_50 to sm_90 (for example A100 or H100). Stage II with the default batch size of 128 needs more than 16 GB of GPU memory; the paper used one 80 GB H100. On a smaller GPU, add `--batch-size 32` (or lower).

```bash
git clone https://github.com/gomes-lab/deep-dos-reasoner.git
cd deep-dos-reasoner
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment.yml
conda activate ddr
python scripts/check_env.py
```

## Benchmarks

Three benchmarks are defined in `data/`. Each has a `manifest.csv` (one sample ID per row) and a `split.pkl` (the train, validation and test row indices into that manifest). The indices are row positions, so the manifests must not be re-sorted.

| Name | `--data_src` | Target | Bins | Range | Materials | Train / Val / Test |
| --- | --- | --- | ---: | --- | ---: | --- |
| `edos` | `binned_dos_128` | eDOS | 128 | −4 to 4 eV around E_F | 37,565 | 30,052 / 3,757 / 3,756 |
| `edos_stable` | `binned_dos_128_stable_only` | eDOS, on-hull materials only | 128 | −4 to 4 eV around E_F | 14,055 | 11,244 / 1,406 / 1,405 |
| `phdos` | `ph_dos_51` | phDOS | 51 | 0 to 1000 cm⁻¹ | 1,524 | 1,220 / 152 / 152 |

The eDOS targets are Materials Project DOS aligned to the Fermi level, smoothed with a 0.5 eV Gaussian and rebinned onto 128 bins with the area preserved; band gaps are kept. The phDOS benchmark is the DFPT phonon dataset of Chen et al. (2021). eDOS IDs are Materials Project IDs; phDOS IDs are the number of the Materials Project ID.

The prepared crystal graphs (one `.chkpt` per material, with its target) are on Hugging Face at [yingheng/DeepDOSReasoner](https://huggingface.co/datasets/yingheng/DeepDOSReasoner), about 200 MB in total. This command downloads them, checks their SHA-256 sums and unpacks them into `data/edos/graphs/` and `data/phdos/graphs/`:

```bash
bash scripts/download_data.sh
```

Pass `edos` or `phdos` to fetch only one of them. `edos_stable` reads the `edos` graphs. If the graphs live elsewhere, a symlink at `data/<name>/graphs` works.

## Training

All commands run from the repository root in the `ddr` environment. Each launcher takes a benchmark name and a seed and uses the settings of the paper's Table S1. Any further arguments go straight to the Python script and take precedence.

- Stage I: 160 epochs, batch size 128, Adam with the inverse-square-root schedule (multiplier 1, 300 warmup steps, decay from epoch 121).
- Stage II: 20 epochs, batch size 128, AdamW (learning rate 1e-3, weight decay 1e-3) with a cosine schedule after a 30% linear warmup, K = 5, `cfl_courant=0.5`.
- Both stages: `d_model=512`, 8 heads, 6 layers per decoder module.

The same settings are used for all three benchmarks. The paper's ensemble uses seeds 1 and 4:

```bash
bash scripts/train_stage1.sh edos 1
```

```bash
bash scripts/train_stage1.sh edos 4
```

```bash
bash scripts/train_stage2.sh edos 1
```

```bash
bash scripts/train_stage2.sh edos 4
```

Stage II with seed `N` starts from the Stage I checkpoint of seed `N`. Checkpoints (best validation MSE) and logs go to `TRAINED/stage1/<benchmark>_stage1_seed<N>/` and `TRAINED/stage2/<benchmark>_stage2_seed<N>/`, and TensorBoard logs to `summary/`. A rerun with the same benchmark and seed replaces the earlier one; use `--save_dir <name>` (and, for Stage II, `--stage1_ckpt <name>`) to keep both. The first visible GPU is used, so choose it with `CUDA_VISIBLE_DEVICES`.

On a Slurm cluster, submit the same launchers with `sbatch` from the repository root. Their headers ask for 1 GPU, 16 CPU cores, 32 GB of RAM and 24 hours; partition, QoS and GPU type go on the command line (`sbatch -p <partition> --qos=<qos> ...`). Logs are written to the repository root as `ddr_stage1-<job id>.out` and so on. Chain the stages with job dependencies:

```bash
s1=$(sbatch --parsable scripts/train_stage1.sh edos 1)
```

```bash
sbatch --dependency=afterok:$s1 scripts/train_stage2.sh edos 1
```

## Evaluation

```bash
bash scripts/test.sh edos 1 4
```

This loads the two Stage I/II pairs, averages their refined spectra with equal weights and scores the test split. The model predicts the normalized shape of the spectrum, so before scoring each prediction is multiplied by the DFT number of states of that material in the energy window. The script prints R² (variance-weighted over bins), MAE, MSE and the Wasserstein distance.

`test.py` only scores benchmark materials. For the eDOS of a new structure (CIF or POSCAR), use the upload box on the [project page](https://gomes-lab.github.io/deepdosreasoner-demo/#upload).

## Paper results

Mean ± standard deviation over five runs on the test split:

| Benchmark | R² ↑ | MAE ↓ | MSE ↓ | WD ↓ |
| --- | --- | --- | --- | --- |
| eDOS | 0.691 ± 0.004 | 2.567 ± 0.002 | 38.142 ± 0.253 | 0.191 ± 0.005 |
| phDOS | 0.688 ± 0.006 | 0.064 ± 0.002 | 0.016 ± 0.001 | 0.065 ± 0.002 |

DeepDOSReasoner predictions for the 91,405 Materials Project entries without a computed eDOS and the 127,768 without a phDOS are at [yingheng/mp-predicted-edos-phdos](https://huggingface.co/datasets/yingheng/mp-predicted-edos-phdos).

## Citation

```bibtex
@article{wang2026deepdosreasoner,
  title   = {DeepDOSReasoner: Physics-grounded reasoning for density-of-states prediction and materials discovery},
  author  = {Wang, Yingheng and Ricci, Francesco and Wang, Zhilong and Yu, Tao and Bai, Junwen and Lei, Xiangyun and Kong, Shufeng and You, Fengqi and Gregoire, John M. and Gomes, Carla P.},
  note    = {Manuscript under review},
  year    = {2026}
}
```

## License

Code: [MIT](LICENSE). Data: CC BY 4.0, derived from the Materials Project and from Chen et al. (2021); the [dataset card](https://huggingface.co/datasets/yingheng/DeepDOSReasoner) lists the works to cite.
