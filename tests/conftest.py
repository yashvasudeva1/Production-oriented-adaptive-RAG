import os

# Isolate test execution from developer .env files so CI and local runs behave identically.
os.environ["OFFLINE_EVAL"] = "1"
os.environ["ALLOW_MOCK_EMBEDDER"] = "1"
os.environ.setdefault("QDRANT_URL", "")
os.environ.setdefault("QDRANT_PATH", "")
os.environ["QDRANT_URL"] = ""
os.environ["QDRANT_PATH"] = ""
