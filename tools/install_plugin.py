"""Copy the plugin (entry file + colorbatch package) into Orca's user plugin folder."""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(os.environ["APPDATA"]) / "OrcaSlicer" / "orca_plugins" / "color_batching"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "orca_plugin" / "color_batching.py", DEST / "color_batching.py")
    pkg = DEST / "colorbatch"
    if pkg.exists():
        shutil.rmtree(pkg)
    shutil.copytree(ROOT / "colorbatch", pkg, ignore=shutil.ignore_patterns("__pycache__"))
    state = DEST / ".install_state.json"
    if not state.exists():  # first install: enabled, same shape Orca writes for local plugins
        state.write_text(json.dumps({
            "capabilities": [{"Color Batching": True}], "enabled": True, "installed_from": "local",
            "installed_version": "0.1.0",
            "permissions": {"events": [], "fs_read": [], "fs_readwrite": [], "full_access": False,
                            "network_http": [], "network_socket": [], "process": [], "threading": False},
            "plugin_name": "Color Batching"}, indent=2), encoding="utf-8")
    print(DEST)


if __name__ == "__main__":
    main()
