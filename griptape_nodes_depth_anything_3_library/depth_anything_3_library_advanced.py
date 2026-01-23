import importlib.metadata
import logging
import subprocess
import sys
from pathlib import Path

import pygit2

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("depth_anything_3_library")


def _clear_cached_modules() -> None:
    """Clear cached modules to ensure library venv packages are used."""
    # Clear importlib.metadata cache
    try:
        if hasattr(importlib.metadata, "_adapters"):
            importlib.metadata._adapters.wrap_mtime_invalidator.cache_clear()
        sys.path_importer_cache.clear()
    except Exception as e:
        logger.warning(f"Could not clear importlib.metadata cache: {e}")

    # Modules that need to be re-imported from library venv
    # Note: Do NOT clear importlib.* modules - they are core Python modules
    prefixes_to_clear = [
        "huggingface_hub",
        "safetensors",
        "transformers",
    ]

    modules_to_clear = [
        module_name
        for module_name in list(sys.modules.keys())
        if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefixes_to_clear)
    ]

    for module_name in modules_to_clear:
        del sys.modules[module_name]
        logger.debug(f"Cleared cached module: {module_name}")


def _get_library_venv_python() -> Path:
    """Get the path to the library venv's Python executable."""
    library_root = Path(__file__).parent
    if sys.platform == "win32":
        venv_python = library_root / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = library_root / ".venv" / "bin" / "python"
    return venv_python


class DepthAnything3LibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for Depth Anything 3."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called before any nodes are loaded from the library."""
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

        # Clear cached modules to ensure library venv packages are used
        _clear_cached_modules()

        logger.info("Initializing depth-anything-3 submodule...")
        depth_anything_path = self._init_depth_anything_submodule()

        self._install_depth_anything(depth_anything_path)

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called after all nodes have been loaded from the library."""
        msg = f"Finished loading nodes for '{library_data.name}' library"
        logger.info(msg)

    def _get_library_root(self) -> Path:
        """Get the library root directory."""
        return Path(__file__).parent

    def _update_submodules_recursive(self, repo_path: Path) -> None:
        """Recursively update and initialize all submodules."""
        repo = pygit2.Repository(str(repo_path))
        repo.submodules.update(init=True)

        for submodule in repo.submodules:
            submodule_path = repo_path / submodule.path
            if submodule_path.exists() and (submodule_path / ".git").exists():
                self._update_submodules_recursive(submodule_path)

    def _init_depth_anything_submodule(self) -> Path:
        """Initialize the depth-anything-3 git submodule."""
        library_root = self._get_library_root()
        depth_anything_submodule_dir = library_root / "depth-anything-3"

        if depth_anything_submodule_dir.exists() and any(depth_anything_submodule_dir.iterdir()):
            logger.info("depth-anything-3 submodule already initialized")
            return depth_anything_submodule_dir

        git_repo_root = library_root.parent
        self._update_submodules_recursive(git_repo_root)

        if not depth_anything_submodule_dir.exists() or not any(depth_anything_submodule_dir.iterdir()):
            raise RuntimeError(
                f"Submodule initialization failed: {depth_anything_submodule_dir} is empty or does not exist"
            )

        logger.info("depth-anything-3 submodule initialized successfully")
        return depth_anything_submodule_dir

    def _get_venv_python_path(self) -> Path:
        """Get the path to the library venv's Python executable."""
        library_root = self._get_library_root()
        if sys.platform == "win32":
            venv_python = library_root / ".venv" / "Scripts" / "python.exe"
        else:
            venv_python = library_root / ".venv" / "bin" / "python"

        if not venv_python.exists():
            raise RuntimeError(f"Library venv Python not found at {venv_python}")
        return venv_python

    def _ensure_pip_installed(self) -> None:
        """Ensure pip is installed in the library venv."""
        venv_python = self._get_venv_python_path()

        result = subprocess.run(
            [str(venv_python), "-m", "pip", "--version"],
            capture_output=True
        )
        if result.returncode == 0:
            logger.info("pip is available in library venv")
            return

        logger.info("pip not found in library venv, installing with ensurepip...")
        subprocess.check_call([
            str(venv_python), "-m", "ensurepip", "--upgrade"
        ])
        logger.info("pip installed successfully")

    def _install_depth_anything(self, depth_anything_path: Path) -> None:
        """Install depth-anything-3 from the submodule into the library venv."""
        venv_python = self._get_venv_python_path()

        self._ensure_pip_installed()

        result = subprocess.run(
            [str(venv_python), "-c", "import depth_anything_3"],
            capture_output=True
        )
        if result.returncode == 0:
            logger.info("depth_anything_3 already installed in library venv")
            return

        logger.info(f"Installing depth_anything_3 from {depth_anything_path}...")
        subprocess.check_call([
            str(venv_python), "-m", "pip", "install",
            str(depth_anything_path)
        ])
        logger.info("depth_anything_3 installed successfully")
