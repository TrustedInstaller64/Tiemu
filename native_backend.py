"""Select the optional native core without changing the reference algorithms."""
import importlib
import warnings

_native = None
_configured = False


def configure_backend(backend="auto"):
    """Select auto/python/cpp; an explicit C++ request must never silently fall back."""
    global _native, _configured
    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("compute backend must be 'auto', 'python', or 'cpp'")
    module = None
    if backend != "python":
        try:
            module = importlib.import_module("_tiemu_native")
        except (ImportError, OSError) as exc:
            message = (
                "Tiemu native C++ extension is unavailable. Build/install the project "
                "with a C++17 compiler, or select the python compute backend. "
                f"Import error: {exc}"
            )
            if backend == "cpp":
                raise RuntimeError(message) from exc
            warnings.warn(message + " Falling back to Python.", RuntimeWarning, stacklevel=2)
    _native = module
    _configured = True
    return module


def get_native():
    """Return the selected extension, or None for the Python reference backend."""
    if not _configured:
        configure_backend("auto")
    return _native
