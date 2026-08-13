#!/usr/bin/env python3
"""Build GraphEm's Sphinx documentation with release-level strictness."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    """Build HTML documentation and fail on every Sphinx warning."""

    root = Path(__file__).resolve().parent
    docs = root / "docs"
    output = docs / "_build" / "html"
    doctrees = docs / "_build" / "doctrees"

    if not docs.is_dir():
        print(f"documentation directory is missing: {docs}", file=sys.stderr)
        return 2

    build_root = docs / "_build"
    if build_root.exists():
        shutil.rmtree(build_root)

    command = [
        "sphinx-build",
        "-W",
        "--keep-going",
        "-n",
        "-b",
        "html",
        "-d",
        str(doctrees),
        str(docs),
        str(output),
    ]
    print("Running:", " ".join(command))
    try:
        subprocess.run(command, cwd=root, check=True)
    except FileNotFoundError:
        print(
            "sphinx-build is unavailable; install with "
            '`python -m pip install -e ".[docs]"`',
            file=sys.stderr,
        )
        return 2
    except subprocess.CalledProcessError as error:
        return error.returncode

    index = output / "index.html"
    if not index.is_file():
        print(f"Sphinx did not create {index}", file=sys.stderr)
        return 1
    print(f"Documentation built successfully: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
