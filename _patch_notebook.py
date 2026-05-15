import json

with open('notebooks/colab_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# ── Cell 1: bump version to v5.0 ──────────────────────────────────────────
src = ''.join(cells[1]['source'])
src = src.replace(
    "NOTEBOOK_VERSION = 'v4.0'  # pure-PyTorch SAGEConv, no torch-geometric",
    "NOTEBOOK_VERSION = 'v5.0'  # DGL backend, no torch-geometric"
)
lines = src.splitlines(keepends=True)
if lines and lines[-1].endswith('\n'):
    lines[-1] = lines[-1].rstrip('\n')
cells[1]['source'] = lines
print('Cell 1 updated.')

# ── Cell 3: add DGL install block ─────────────────────────────────────────
src3 = ''.join(cells[3]['source'])
OLD3 = "        'uni2ts', 'gluonts', 'arch'])\n    _pin_torchvision()   # pin after packages may have changed torch"
NEW3 = (
    "        'uni2ts', 'gluonts', 'arch'])\n"
    "    # DGL — auto-detect CPU vs GPU (no torchvision dependency)\n"
    "    import torch as _t\n"
    "    _cv = getattr(getattr(_t, 'version', None), 'cuda', None)\n"
    "    _dgl_extra = ['-f', 'https://data.dgl.ai/wheels/cu' + _cv.replace('.','')[:3] + '/repo.html'] if _cv else []\n"
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'dgl'] + _dgl_extra)\n"
    "    _pin_torchvision()   # pin after packages may have changed torch"
)
if OLD3 in src3:
    src3 = src3.replace(OLD3, NEW3)
    lines3 = src3.splitlines(keepends=True)
    if lines3 and lines3[-1].endswith('\n'):
        lines3[-1] = lines3[-1].rstrip('\n')
    cells[3]['source'] = lines3
    print('Cell 3 updated.')
else:
    print('Cell 3: WARNING block not found. Checking raw...')
    idx = src3.find("'uni2ts'")
    print(repr(src3[max(0, idx-60):idx+80]))

# ── Cell 8: fix inference call ────────────────────────────────────────────
src8 = ''.join(cells[8]['source'])
OLD8 = 'gnn_model(feats, g.edge_index.to(device)).cpu().numpy().ravel()'
NEW8 = 'gnn_model(g.g.to(device), feats).cpu().numpy().ravel()'
if OLD8 in src8:
    src8 = src8.replace(OLD8, NEW8)
    lines8 = src8.splitlines(keepends=True)
    if lines8 and lines8[-1].endswith('\n'):
        lines8[-1] = lines8[-1].rstrip('\n')
    cells[8]['source'] = lines8
    print('Cell 8 updated.')
else:
    print('Cell 8: WARNING inference line not found.')
    idx = src8.find('gnn_model(')
    print(repr(src8[idx:idx+80]))

with open('notebooks/colab_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Notebook saved OK.')
