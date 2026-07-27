import io
import json
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

result = subprocess.run(
    [
        "curl.exe",
        "-s",
        "https://api.github.com/repos/pablohuidobrogarcia/An-lisis-de-sentimiento-de-los-aficionadosMundial-2026-/actions/workflows/daily_collection.yml/runs?per_page=10",
    ],
    capture_output=True,
    text=True,
)
data = json.loads(result.stdout)
for r in data.get("workflow_runs", []):
    created = r["created_at"][:19]
    status = r["status"]
    conclusion = r["conclusion"] or "-"
    title = r.get("display_title", "")[:60]
    print(f"{created} | {status:>8} | {conclusion:>10} | {title}")

print(f'\nTotal runs returned: {len(data.get("workflow_runs", []))}')
