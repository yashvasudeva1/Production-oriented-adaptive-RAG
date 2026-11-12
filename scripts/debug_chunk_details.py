import sys
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

sys.stdout.reconfigure(encoding='utf-8')

with open('metadata/debug_chunks_1706.03762v7.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

for idx in [21, 22, 23, 24, 45, 46, 47]:
    if idx < len(chunks):
        c = chunks[idx]
        print(f"=== Chunk {idx} (ID: {c['id']}, Page: {c['page']}, Type: {c['chunk_type']}) ===")
        print(c['text'])
        print('-' * 60)
