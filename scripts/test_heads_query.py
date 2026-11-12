import os
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

os.environ["OFFLINE_EVAL"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

from src.components.orchestration.pipeline import RAGOrchestrator

def main():
    orchestrator = RAGOrchestrator()
    query = "How many attention heads does the base model use?"
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    # Test in adaptive mode
    print("\n--- 1. Testing in ADAPTIVE mode ---")
    res_adapt = orchestrator.query(query, mode="adaptive")
    print(f"Execution mode: {res_adapt.execution_mode}")
    print(f"Cascade stages: {res_adapt.retrieval.cascade_stages_executed}")
    print(f"Abstained: {res_adapt.abstained}")
    print(f"Confidence: {res_adapt.confidence}")
    print(f"Answer: {res_adapt.answer}")
    print(f"Sources ({len(res_adapt.sources)}):")
    for i, s in enumerate(res_adapt.sources, 1):
        print(f"  [{i}] {s.get('chunk_id')} (p.{s.get('page')}, {s.get('section')}): {s.get('snippet')[:100]}...")

    # Test in fast mode
    print("\n--- 2. Testing in FAST mode ---")
    res_fast = orchestrator.query(query, mode="fast")
    print(f"Execution mode: {res_fast.execution_mode}")
    print(f"Cascade stages: {res_fast.retrieval.cascade_stages_executed}")
    print(f"Abstained: {res_fast.abstained}")
    print(f"Confidence: {res_fast.confidence}")
    print(f"Answer: {res_fast.answer}")
    print(f"Sources ({len(res_fast.sources)}):")
    for i, s in enumerate(res_fast.sources, 1):
        print(f"  [{i}] {s.get('chunk_id')} (p.{s.get('page')}, {s.get('section')}): {s.get('snippet')[:100]}...")

if __name__ == "__main__":
    main()
