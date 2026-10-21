# AI-generated code.
"""Validate release metadata shared by packaging and citation files."""

from configparser import ConfigParser
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def read_package_version() -> str:
    """Read the package version from setup.cfg."""
    config = ConfigParser()
    config.read(ROOT / "setup.cfg", encoding="utf-8")
    return config["metadata"]["version"].strip()


def read_citation_version() -> str:
    """Read the top-level version from CITATION.cff."""
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*([^\s#]+)\s*$", citation, re.MULTILINE)
    if match is None:
        raise ValueError("CITATION.cff is missing a top-level version field")
    return match.group(1)


def main() -> None:
    """Fail when package and citation versions differ."""
    package_version = read_package_version()
    citation_version = read_citation_version()
    if package_version != citation_version:
        raise SystemExit(
            f"Version mismatch: setup.cfg={package_version}, "
            f"CITATION.cff={citation_version}"
        )
    print(f"Release metadata versions match: {package_version}")


if __name__ == "__main__":
    main()
