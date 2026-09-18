#!/usr/bin/env python3
"""CLI shim for the site recorder. Implementation lives in src/perimeter/record.py.

Usable without installing the package:
    python tools/record_site.py --source rtsp://... --outdir recordings
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from perimeter.record import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
