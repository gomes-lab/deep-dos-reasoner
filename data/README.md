# Data

Each benchmark directory holds its definition, tracked in git:

```text
data/<benchmark>/
  manifest.csv     sample IDs, one per row (column "mpid"); split indices refer to these rows
  split.pkl        fixed (train, val, test) row indices into manifest.csv
```

| Directory | Spectrum | Samples | Train / Val / Test |
| --- | --- | ---: | --- |
| `edos` | Electronic DOS | 37,565 | 30,052 / 3,757 / 3,756 |
| `edos_stable` | Electronic DOS, stable-only subset (energy above hull = 0) | 14,055 | 11,244 / 1,406 / 1,405 |
| `phdos` | Phonon DOS | 1,524 | 1,220 / 152 / 152 |

The prepared crystal graphs are added by

```bash
bash scripts/download_data.sh
```

which fetches them from
[huggingface.co/datasets/yingheng/DeepDOSReasoner](https://huggingface.co/datasets/yingheng/DeepDOSReasoner),
checks their SHA-256 sums and unpacks them to `data/edos/graphs/` and `data/phdos/graphs/`.
`edos_stable` uses the `edos` graphs.
