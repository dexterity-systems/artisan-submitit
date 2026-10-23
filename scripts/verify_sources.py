"""Verify that qualification imports the recorded sibling source checkouts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import artisan

import artisan_submitit


def main() -> None:
    """Check imported paths and the core revision before executing tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("core_checkout", type=Path)
    parser.add_argument("provider_checkout", type=Path)
    parser.add_argument("core_revision")
    arguments = parser.parse_args()
    evidence = {}
    for name, module, checkout in (
        ("artisan", artisan, arguments.core_checkout),
        ("artisan-submitit", artisan_submitit, arguments.provider_checkout),
    ):
        source = Path(module.__file__).resolve()
        assert source.is_relative_to(checkout.resolve() / "src"), source
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        evidence[name] = {"revision": revision, "import_path": str(source)}
    assert evidence["artisan"]["revision"] == arguments.core_revision, evidence
    sys.stdout.write(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
