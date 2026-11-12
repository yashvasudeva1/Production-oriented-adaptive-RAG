import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._faithfulness import Faithfulness

from src.components.orchestration import RAGOrchestrator
from src.evaluation.ragas_dataset import RagasDatasetBuilder
from src.evaluation.ragas_metrics import DeterministicEvaluatorEngine


def main():
    print("=" * 80)
    print("RUNNING OFFICIAL RAGAS EVALUATION VALIDATION ON 5 TEST CASES")
    print("=" * 80)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[ERROR] GROQ_API_KEY required for official RAGAS evaluation.")
        sys.exit(1)

    # 1. Initialize official RAGAS LLM & Embeddings
    print("\n[1/4] Initializing RAGAS Judge (qwen/qwen3.8-27b on Groq)...")
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
    ragas_llm = llm_factory("qwen/qwen3.8-27b", client=client)
    if hasattr(ragas_llm, "model_args") and isinstance(ragas_llm.model_args, dict):
        ragas_llm.model_args["max_tokens"] = 850  # Enforce Groq on-demand OTPM limit (< 1000) while allowing full output

    print("[2/4] Initializing local HuggingFace embeddings...")
    hf_emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True},
    )
    ragas_emb = LangchainEmbeddingsWrapper(hf_emb)

    # 2. Initialize orchestrator and dataset
    print("[3/4] Initializing RAG Orchestrator and Dataset...")
    builder = RagasDatasetBuilder()
    orchestrator = RAGOrchestrator()

    # Synchronize Qdrant if in-memory
    if orchestrator.qdrant.count() == 0 and orchestrator.bm25.count() > 0:
        from src.components.chunking.models import Chunk
        loaded_chunks = [Chunk.from_dict(c) for c in orchestrator.bm25._all_chunks]
        vecs = orchestrator.embedder.embed_texts([c.text for c in loaded_chunks])
        orchestrator.qdrant.upsert_chunks(loaded_chunks, vecs)

    # Select 5 distinct validation cases covering different categories
    target_ids = ["exact_01", "exact_02", "conc_01", "comp_01", "hop_01"]
    cases = [c for c in builder.cases if c.query_id in target_ids]

    samples = []
    deterministic_records = []
    case_details = []

    print(f"\n[4/4] Executing RAG pipeline and scoring {len(cases)} cases...")
    for idx, case in enumerate(cases, start=1):
        print(f"  [{idx}/{len(cases)}] Query {case.query_id} ({case.category}): '{case.query}'")
        resp = orchestrator.query(case.query)
        ref = builder.get_ground_truth(case.query_id)
        contexts = [s.get("text", "") for s in resp.sources]

        # Official RAGAS sample
        sample = SingleTurnSample(
            user_input=case.query,
            response=resp.answer,
            reference=ref,
            retrieved_contexts=contexts,
        )
        samples.append(sample)

        # Deterministic offline scoring
        det_faith = DeterministicEvaluatorEngine.evaluate_faithfulness(resp.answer, contexts)
        det_corr = DeterministicEvaluatorEngine.evaluate_answer_correctness(resp.answer, ref)
        det_prec = DeterministicEvaluatorEngine.evaluate_context_precision(case.query, contexts, ref)
        det_rec = DeterministicEvaluatorEngine.evaluate_context_recall(case.query, contexts, ref)

        deterministic_records.append({
            "query_id": case.query_id,
            "deterministic_faithfulness": round(det_faith, 4),
            "deterministic_token_f1": round(det_corr, 4),
            "deterministic_context_precision": round(det_prec, 4),
            "deterministic_context_recall": round(det_rec, 4),
        })

        case_details.append({
            "query_id": case.query_id,
            "category": case.category,
            "query": case.query,
            "reference": ref,
            "response": resp.answer,
            "contexts_count": len(contexts),
            "deterministic_token_f1": round(det_corr, 4),
            "deterministic_faithfulness": round(det_faith, 4),
        })
        time.sleep(1.0)  # Gentle spacing to avoid hitting RPM rate limits

    # Evaluate official RAGAS metrics with serialized execution (max_workers=1) to respect Groq rate limits
    print("\nRunning official RAGAS judge (AnswerCorrectness + Faithfulness)...")
    from ragas.run_config import RunConfig
    run_cfg = RunConfig(max_workers=1, timeout=90, max_retries=5, max_wait=60)
    eval_dataset = EvaluationDataset(samples=samples)
    metrics_to_run = [AnswerCorrectness(), Faithfulness()]
    ragas_results = evaluate(
        dataset=eval_dataset,
        metrics=metrics_to_run,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_cfg,
    )

    print("\n" + "=" * 80)
    print("OFFICIAL RAGAS VS DETERMINISTIC FALLBACK COMPARISON")
    print("=" * 80)
    print(f"{'Query ID':<18} | {'Official Correctness':<20} | {'Deterministic Token F1':<22} | {'Official Faith':<15} | {'Det Faith':<10}")
    print("-" * 92)

    df_results = ragas_results.to_pandas()
    comparison_summary = []

    for i, row in df_results.iterrows():
        qid = target_ids[i]
        off_corr = float(row.get("answer_correctness", 0.0))
        off_faith = float(row.get("faithfulness", 0.0))
        det_rec = deterministic_records[i]
        det_f1 = det_rec["deterministic_token_f1"]
        det_faith = det_rec["deterministic_faithfulness"]

        print(f"{qid:<18} | {off_corr:<20.4f} | {det_f1:<22.4f} | {off_faith:<15.4f} | {det_faith:<10.4f}")

        comparison_summary.append({
            "query_id": qid,
            "official_answer_correctness": round(off_corr, 4),
            "deterministic_token_f1": det_f1,
            "official_faithfulness": round(off_faith, 4),
            "deterministic_faithfulness": det_faith,
            "response": case_details[i]["response"],
            "reference": case_details[i]["reference"],
        })

    out_file = Path("metadata/ragas_validation_subset.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    print(f"\nSaved detailed comparison to: {out_file}")


if __name__ == "__main__":
    main()
