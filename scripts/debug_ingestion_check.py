import sys
import json
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

from src.components.ingestion.pipeline import IngestionPipeline
from src.components.chunking.pipeline import ChunkingPipeline

def run_ingestion_check():
    # Test both possible PDF filenames
    for filename in ['1706.03762v7.pdf', 'NIPS-2017-attention-is-all-you-need-Paper.pdf']:
        pdf_path = Path('documents') / filename
        if not pdf_path.exists():
            continue
        print(f"\n=======================================================")
        print(f"PROCESSING FILE: {filename}")
        print(f"=======================================================")
        
        ingest_pipe = IngestionPipeline()
        chunk_pipe = ChunkingPipeline()

        doc = ingest_pipe.ingest_file(pdf_path)
        print(f"Document loaded: {doc.filename}, ID: {doc.document_id}, blocks: {len(doc.blocks)}")

        chunks = chunk_pipe.chunk_document(doc)
        print(f"Total chunks generated: {len(chunks)}")

        dump_path = Path(f"metadata/debug_chunks_{pdf_path.stem}.json")
        dump_data = []
        for c in chunks:
            dump_data.append({
                "id": c.chunk_id,
                "chunk_type": c.chunk_type,
                "page": c.page,
                "text": c.text,
                "section": c.section,
                "parent_id": c.parent_id
            })

        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)
        print(f"Dumped chunks to: {dump_path}")

        targets = ["h = 8", "h=8", "8 parallel", "parallel attention layers", "d_k", "64"]
        found_map = {t: [] for t in targets}

        for idx, c in enumerate(chunks):
            for t in targets:
                if t.lower() in c.text.lower():
                    found_map[t].append({
                        "chunk_index": idx,
                        "chunk_id": c.chunk_id,
                        "page": c.page,
                        "chunk_type": c.chunk_type,
                        "snippet": c.text[:120].replace('\n', ' ')
                    })

        print("\nTarget Search Results:")
        for t, hits in found_map.items():
            print(f"\n  Keyword: '{t}' (found in {len(hits)} chunks):")
            for h in hits:
                print(f"    - Index {h['chunk_index']} | ID: {h['chunk_id']} | Page: {h['page']} | Type: {h['chunk_type']}")
                print(f"      Text: {h['snippet']}...")

if __name__ == "__main__":
    run_ingestion_check()
