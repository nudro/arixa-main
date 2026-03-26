"""Isolate test database and upload directory before any ``app`` imports."""

from __future__ import annotations

import os
import tempfile

# Must run at import time so ``app.db.session`` binds to these values on first load.
_fd, _test_db_path = tempfile.mkstemp(suffix=".sqlite")
os.close(_fd)
os.environ["ARIXA_DATABASE_URL"] = f"sqlite:///{_test_db_path}"
os.environ["ARIXA_UPLOAD_ROOT"] = tempfile.mkdtemp(prefix="arixa_uploads_")
