import sys as _sys
import types as _types

# Colab's kernel partially imports torchvision at startup, leaving it in
# sys.modules without the 'extension' attribute.  torch_geometric then imports
# torchvision._meta_registrations which calls torchvision.extension._has_ops()
# before extension is ready → AttributeError.
#
# Fix: inject a stub extension (returning False) so _has_ops() is safe.
# Do NOT clear sys.modules and reimport — the C extension (.so) is already
# loaded and operator registrations cannot be safely re-run.
_tv = _sys.modules.get("torchvision")
if _tv is not None and not hasattr(_tv, "extension"):
    _stub = _types.SimpleNamespace(_has_ops=lambda: False)
    _tv.extension = _stub
    _sys.modules.setdefault("torchvision.extension", _stub)
    del _stub
elif _tv is None:
    # torchvision not yet touched — import it fully now so torch_geometric
    # never sees an uninitialized module.
    try:
        import torchvision  # noqa: F401
    except Exception:
        pass

del _sys, _types, _tv
