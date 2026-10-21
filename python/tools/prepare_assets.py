"""Prepare external XWIS assets for VERA's explicit multiplayer media mode."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

SOURCES = {
    "yuri": "https://web.archive.org/web/20170612013942id_/http://xwis.net/downloads/Yuris-Revenge-Multiplayer.exe",
    "ra2": "https://web.archive.org/web/20130610123451id_/http://xwis.net/downloads/Red-Alert-2-Multiplayer.exe",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    downloads = root / "downloads"
    downloads.mkdir(exist_ok=True)
    game = root / "game"
    game.mkdir(exist_ok=True)
    manifest = {"media_mode": "Multiplayer", "sources": {}, "omitted_empty_archives": []}
    for name, url in SOURCES.items():
        archive = downloads / f"{name}.exe"
        if not archive.exists():
            partial = archive.with_suffix(".partial")
            urllib.request.urlretrieve(url, partial)
            partial.rename(archive)
        extracted = root / f"extract-{name}"
        subprocess.run(["7z", "x", "-y", str(archive), f"-o{extracted}"], check=True, stdout=subprocess.DEVNULL)
        manifest["sources"][name] = {"url": url, "bytes": archive.stat().st_size}
        for asset in extracted.glob("*.mix"):
            if asset.stat().st_size == 0:
                manifest["omitted_empty_archives"].append(f"{name}/{asset.name}")
            else:
                target = game / asset.name
                if target.exists() and target.stat().st_size != asset.stat().st_size:
                    raise RuntimeError(f"conflicting asset: {target}")
                shutil.copyfile(asset, target)
    (root / "sources.json").write_text(json.dumps(manifest, indent=2))
    print(game)


if __name__ == "__main__":
    main()
