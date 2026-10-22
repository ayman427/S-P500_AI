"""Generate a server-only secret and install it into the selected Convex deployment."""
from pathlib import Path
import secrets
import subprocess
import tempfile

root = Path(__file__).resolve().parent
environment = root / ".env.local"
if not environment.exists():
    raise SystemExit("Run npm run convex:dev first to choose a deployment")
text = environment.read_text(encoding="utf-8")
existing = next((line.split("=", 1)[1] for line in text.splitlines()
                 if line.startswith("PREDICTION_STORE_SECRET=")), None)
secret = existing or secrets.token_hex(32)
# The CLI reads the secret from a temporary file; it is not a process argument.
with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False, dir=root) as handle:
    handle.write(f"PREDICTION_STORE_SECRET={secret}\n")
    temporary = Path(handle.name)
try:
    result = subprocess.run(["node", "node_modules/convex/bin/main.js", "env", "set", "--from-file", str(temporary)],
                            cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit("Could not configure Convex secret. Make sure npm run convex:dev is running.")
    if not existing:
        with environment.open("a", encoding="utf-8") as handle:
            handle.write(f"\nPREDICTION_STORE_SECRET={secret}\n")
    print("Convex and Python now share a server-only secret. Restart the Python API.")
finally:
    temporary.unlink(missing_ok=True)
