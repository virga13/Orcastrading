import sys, io
from dotenv import load_dotenv
load_dotenv()

# Windows Terminal uses UTF-8 but Python defaults to CP1252 on Windows.
# Reconfigure stdout/stderr to UTF-8 before any Rich output is created.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from p4_live.cli import main
main()
