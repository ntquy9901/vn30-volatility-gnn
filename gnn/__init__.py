import sys as _sys
import os as _os
import types as _types

# ── Strategy 1: patch _meta_registrations.py (root-cause fix) ─────────────────
# Some Colab torchvision builds import _meta_registrations before 'extension'
# is assigned, causing AttributeError on 'import torchvision'.
# Patch the guard to use getattr so the check is safe during partial init.
for _p in _sys.path:
    _f = _os.path.join(_p, 'torchvision', '_meta_registrations.py')
    if _os.path.isfile(_f):
        try:
            _src = open(_f, encoding='utf-8').read()
            _bad = 'if torchvision.extension._has_ops():'
            _fix = ('if getattr(torchvision, "extension", None) is not None'
                    ' and torchvision.extension._has_ops():')
            if _bad in _src:
                open(_f, 'w', encoding='utf-8').write(_src.replace(_bad, _fix))
        except Exception:
            pass
        break

# ── Strategy 2: stub fallback if module already partial ───────────────────────
_tv = _sys.modules.get("torchvision")
if _tv is not None and not hasattr(_tv, "extension"):
    _stub = _types.SimpleNamespace(_has_ops=lambda: False)
    _tv.extension = _stub
    _sys.modules.setdefault("torchvision.extension", _stub)

del _sys, _os, _types
