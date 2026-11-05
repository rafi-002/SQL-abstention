"""Print OpenRouter model ids and prices (USD per million tokens) matching a substring. Public endpoint, no key."""
import json
import sys
import urllib.request

pat = sys.argv[1] if len(sys.argv) > 1 else "anthropic/"
with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as r:
    models = json.load(r)["data"]
for m in models:
    if pat in m["id"]:
        p = m.get("pricing", {})
        print(m["id"], "in", round(float(p.get("prompt", 0)) * 1e6, 3), "out", round(float(p.get("completion", 0)) * 1e6, 3),
              "ctx", m.get("context_length"))
