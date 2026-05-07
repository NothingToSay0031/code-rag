MCP smoke suite
---------------
Run:
  set PYTHONPATH=src
  pytest -q tests\mcp_smoke\test_mcp_smoke.py

Optional env vars:
  MCP_SMOKE_FORCE_REINDEX=1
  MCP_SMOKE_ARTIFACT_ROOT=tests\artifacts\mcp-smoke

Artifacts:
  calls\<tool>.json   (structured metadata + params + escaped output)
  calls\<tool>.txt    (raw output, human-readable newlines)
