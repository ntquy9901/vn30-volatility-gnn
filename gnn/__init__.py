import sys as _sys

# Colab starts with torchvision partially initialized; torch_geometric then
# imports torchvision._meta_registrations which calls torchvision.extension
# before extension is ready → AttributeError. Evict the partial module so a
# clean re-import runs before any torch_geometric code loads.
_partial = not hasattr(_sys.modules.get("torchvision"), "extension")
if _partial:
    for _k in [k for k in list(_sys.modules) if k.startswith("torchvision")]:
        del _sys.modules[_k]
    import torchvision
    import torchvision.extension  # noqa: F401

del _sys, _partial
