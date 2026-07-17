import os
import sys
from pathlib import Path
import site


def bootstrap_cuda_dll_path() -> None:
    """Add pip-installed NVIDIA DLL directories on Windows."""
    if sys.platform != "win32":
        return

    candidate_nvidia_roots = []

    # Prefer roots from the active interpreter to avoid hardcoded venv paths.
    for site_dir in site.getsitepackages():
        candidate_nvidia_roots.append(Path(site_dir) / "nvidia")

    user_site = site.getusersitepackages()
    if user_site:
        candidate_nvidia_roots.append(Path(user_site) / "nvidia")

    # Fallback for this repository layout.
    project_root = Path(__file__).resolve().parents[1]
    candidate_nvidia_roots.append(project_root / "venv" / "Lib" / "site-packages" / "nvidia")

    dll_dirs = []
    for nvidia_root in candidate_nvidia_roots:
        for rel in ("cublas/bin", "cudnn/bin", "cuda_runtime/bin"):
            dll_dir = nvidia_root / rel
            if dll_dir.is_dir() and dll_dir not in dll_dirs:
                dll_dirs.append(dll_dir)

    # Add directories via loader API when available.
    for dll_dir in dll_dirs:
        os.add_dll_directory(str(dll_dir))

    # Also prepend PATH to handle native dependency chains in some loaders.
    if dll_dirs:
        existing = os.environ.get("PATH", "")
        prepend = ";".join(str(p) for p in dll_dirs)
        os.environ["PATH"] = prepend + (";" + existing if existing else "")
