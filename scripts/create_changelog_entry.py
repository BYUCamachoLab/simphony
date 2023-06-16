#!/usr/bin/env python3

"""
This script will create a changelog entry for the specified version. Usually
called by the makefile when creating a new version.

Usage:
    python scripts/create_changelog_entry.py VERSION

Example:
    python scripts/create_changelog_entry.py 0.7.0
"""

import sys
from pathlib import Path

from packaging.version import parse

# Get first command line argument
version = parse(sys.argv[1])

template = f"""---

## [{str(version)}](https://github.com/BYUCamachoLab/simphony/tree/v{str(version)}) - <small>YYYY-MM-DD</small>

### Added
- N/A

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

---
"""

CHANGELOG = Path(__file__).parent.parent / "CHANGELOG.md"
with CHANGELOG.open() as f:
    text = f.read()

# Split changelog by horizontal rule
entries = text.split("\n---\n")
entries = [entry.strip() for entry in entries]

# Insert the template at the top of the changelog
with open("temp.md", "w", encoding="utf8") as f:
    f.write(f"{entries[0]}\n\n")
    f.write(f"{template}\n")
    f.write("\n\n---\n\n".join(entries[1:]))
    f.write("\n")

print(f"Created entry for version {str(version)}. Don't forget to update the date in the changelog entry!")
