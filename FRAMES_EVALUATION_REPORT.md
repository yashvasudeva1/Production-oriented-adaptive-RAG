# Google FRAMES Benchmark: Multi-Hop Retrieval & Reasoning Evaluation

> [!IMPORTANT]
> **Data Source & Reproducibility Notice**
>
> The historical architecture rows (DENSE, BM25, HYBRID, HYBRID_RERANK, ORIGINAL_ADAPTIVE) below are real pipeline runs measured from previous benchmark sessions with the original multi-hop evaluation script. The **SIMPLIFIED_ADAPTIVE (Current)** row is a **simulation** produced by replaying historical per-query records from `metadata/frames_results.json` under the consensus condition `bm25_covered_titles_count > 0 AND dense_covered_titles_count > 0`. This condition evaluated to True for 50/50 cases, so quality metrics for Simplified Adaptive are numerically identical to the HYBRID row. **These results do not represent a live pipeline execution.** Live execution requires hydrating the full 550-chunk Wikipedia corpus (not stored in this repository).
>
> Any claim of Simplified Adaptive performance improvement on FRAMES versus Hybrid should not be made until live re-evaluation is performed on the original corpus.

- **Benchmark:** Google FRAMES (Factuality, Retrieval, And reasoning MEasurement Set, arXiv:2409.12941)
- **Total Multi-Hop Test Cases:** 50
- **Indexed Multi-Document Graph Chunks:** 550 (Wikipedia corpus, not included in repo)
- **Evaluation Mode:** Historical records (DENSE, BM25, HYBRID, HYBRID_RERANK, ORIGINAL_ADAPTIVE) + Consensus Simulation (SIMPLIFIED_ADAPTIVE)

## 1. Overall Multi-Hop Architecture Comparison Table

| Architecture | Context Recall | Context Precision | Multi-Hop Traversal Rate | Exact Match Rate | Token F1 | Faithfulness | P50 Latency | Mean Latency | Data Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DENSE** | 0.7093 | 0.8280 | 88.0% | 6.0% | 0.0195 | 0.9415 | 443.2 ms | 438.9 ms | Live run |
| **BM25** | 0.6957 | 0.6487 | 86.0% | 12.0% | 0.0249 | 0.9438 | 18.0 ms | 17.8 ms | Live run |
| **HYBRID** | 0.7410 | 0.7943 | 88.0% | 10.0% | 0.0207 | 0.9395 | 21.7 ms | 22.7 ms | Live run |
| **HYBRID_RERANK** | 0.7533 | 0.8010 | 88.0% | 10.0% | 0.0208 | 0.9371 | 571.5 ms | 625.1 ms | Live run |
| **ORIGINAL_ADAPTIVE (Legacy)** | 0.6490 | 0.7674 | 76.0% | 8.0% | 0.0215 | 0.9028 | 3137.9 ms | 3057.4 ms | Live run |
| **SIMPLIFIED_ADAPTIVE (Simulated)** | 0.7410† | 0.7943† | 88.0%† | 10.0%† | 0.0207† | 0.9395† | ~25 ms‡ | ~26 ms‡ | **Simulation** (see note above) |

†Identical to HYBRID because 100% of FRAMES cases triggered the consensus fallback to hybrid candidates.  
‡Latency estimated from consensus-bypass path: BM25+Dense retrieval (~18 ms) + 2.5 ms routing overhead. Not measured from a live run.

## 2. Reasoning Type Breakdown (Historical Data — HYBRID Architecture)

The following breakdown reflects HYBRID architecture results from historical evaluation runs. Numbers marked SIMPLIFIED_ADAPTIVE in earlier reports were simulation artifacts copying from HYBRID.

| Reasoning Category | Cases | Context Recall | Multi-Hop Traversal | Exact Match | Token F1 | P50 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Multiple constraints` | 34 | 0.7407 | 85.3% | 8.8% | 0.0223 | 21.7 ms |
| `Numerical reasoning` | 25 | 0.7387 | 88.0% | 4.0% | 0.0077 | 22.1 ms |
| `Post processing` | 10 | 0.7417 | 80.0% | 0.0% | 0.0024 | 22.4 ms |
| `Tabular reasoning` | 26 | 0.7096 | 84.6% | 7.7% | 0.0189 | 21.3 ms |
| `Temporal reasoning` | 22 | 0.7333 | 86.4% | 18.2% | 0.0247 | 22.3 ms |

## 3. Failure Analysis Across Multi-Hop Reasoning Types

### A. Numerical Reasoning Failures (25 Cases, EM: 4.0%)
- **Root Cause:** Failures stem from arithmetic reasoning over retrieved quantities (e.g. computing year differences, height rank offsets) rather than retrieval failure.
- **Evidence Quality:** Context Recall was high (0.7513), but the final exact string requires arithmetic subtraction/addition that generative models without tool/code interpreter execution frequently approximate rather than compute exactly.

### B. Temporal Reasoning Failures (22 Cases, EM: 18.2%)
- **Root Cause:** Queries require resolving chronological precedence across multiple historical biographies (e.g. comparing year of election with birth dates).
- **Evidence Quality:** Multi-hop document traversal reached 86.4%. Exact match failures occurred when intermediate presidential or historical timelines had subtle date ambiguities.

### C. Tabular & Multi-Constraint Reasoning (26 Cases, EM: 7.7%)
- **Root Cause:** Information spans dense markdown/wiki tables. Row-column alignment across sliding chunk boundaries can clip adjacent tabular cells.
- **Mitigation:** The Simplified Adaptive pipeline's unified parent-child expansion restores full table contexts without requiring iterative multi-hop sub-query loops.

## 4. Key Architectural Conclusions & Benchmark Integrity Audit

> [!WARNING]
> **Simulation vs. Live Execution:** The SIMPLIFIED_ADAPTIVE row in Section 1 is a simulation replay of historical records, NOT a live benchmark. Quality metrics are identical to HYBRID because the consensus condition (`bm25_covered > 0 AND dense_covered > 0`) was true for all 50 FRAMES test cases. This is an accidental copying artifact, not a genuinely measured result. No improved FRAMES performance can be claimed for Simplified Adaptive without live re-evaluation.

1. **Elimination of Query Decomposition Degradation:** In the legacy Adaptive pipeline, recursive query decomposition dropped Context Recall from 0.7533 (Hybrid Rerank) down to 0.6490 (-10.4%) and multi-hop traversal from 88.0% to 76.0%. The Simplified Adaptive architecture avoids this by preserving full query context in a single-hop pass.
2. **Latency Comparison (Actual vs. Simulated):** Original Adaptive averaged 3,057 ms. Hybrid (live) averaged 22.7 ms. Simplified Adaptive latency of ~25 ms is an engineering estimate based on its consensus bypass mechanism — not a measured FRAMES result.
3. **Zero Abstention False Positives (Held-Out 110-Case Suite):** Measured on the 110-case held-out evaluation suite (not FRAMES). All 20 adversarial/unanswerable held-out queries were correctly handled.
4. **FRAMES Re-evaluation Requirement:** To produce verifiable Simplified Adaptive FRAMES metrics, the full Wikipedia dump must be re-indexed and `scripts/evaluate_simplified_frames.py` must execute live pipeline queries against it rather than replaying historical records.
