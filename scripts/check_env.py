"""Report the installed stack, the GPU and the downloaded graphs.

    python scripts/check_env.py
"""
import os

import pandas
import pymatgen.core
import scipy
import sklearn
import torch
import torch_geometric
import torch_scatter

print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'PyG {torch_geometric.__version__}, torch-scatter {torch_scatter.__version__}')
print(f'pymatgen {pymatgen.core.__version__}, scikit-learn {sklearn.__version__}, '
      f'SciPy {scipy.__version__}, pandas {pandas.__version__}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}, {props.total_memory / 2**30:.0f} GB')
else:
    print('GPU: none visible (training and evaluation need one)')

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
for name in ('edos', 'phdos'):
    graphs = os.path.join(root, name, 'graphs')
    n = len(os.listdir(graphs)) if os.path.isdir(graphs) else 0
    print(f'data/{name}/graphs: {n} files' + ('' if n else ' (run: bash scripts/download_data.sh)'))
