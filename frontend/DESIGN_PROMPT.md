# Frontend design / implementation prompt

Design and implement a premium, light-themed web frontend for **Production-oriented Adaptive RAG** (the codebase in this repository). Treat the product as an evidence-first research workspace for engineers, not a generic chatbot or AI landing page.

## Visual direction

Use the restraint and material clarity of Apple software: soft neutral canvas (#f5f5f7), white working surfaces, near-black typography, subtle hairline borders, restrained cobalt-blue primary actions, muted grays, and small semantic green/amber statuses. Use highly legible modern system-like typography (Inter is acceptable), carefully aligned baselines, deliberate whitespace, compact labels, and calm transitions. Surfaces should be nearly flat, with only subtle elevation where hierarchy requires it. Make it feel considered, confident, precise, and fast.

Avoid all visual clichés associated with generic AI templates: no purple/pink gradients, neon accents, glowing blobs, glassmorphism for its own sake, robot imagery, floating glowing orbs, abstract AI stock illustrations, oversized decorative dashboard cards, unnecessary rounded-card grids, or meaningless charts. No ornamental animation. Each element must clarify state, hierarchy, evidence, or an action.

## Product and backend understanding

Build around the actual code and API capabilities in `Production-oriented-adaptive-RAG`:
- FastAPI service, with `GET /health` and `GET /metrics`.
- `POST /query/stream` emits retrieval metadata, answer deltas, and a final structured response with sources, confidence, latency, citation validation, and retrieval telemetry.
- `POST /query` for non-streamed answers.
- `POST /search` for dense, BM25 keyword, and hybrid candidate retrieval.
- Adaptive routing among fast, balanced, and deep paths; hybrid dense+BM25 retrieval; rank fusion; confidence estimation; conditional cross-encoder reranking; parent-child expansion; evidence gate/abstention; grounded generation and citation validation.
- Document upload/list operations where available.

Do not invent backend features or call benchmark results live. Gracefully tolerate API absence and make the API base URL configurable. Keep demo/preview data visibly labelled as preview or recorded benchmark data.

## Information architecture

Create four clear destinations, reachable via a stable sidebar and responsive mobile navigation:

1. **Query workspace** - a focused query composer with mode selector (Adaptive, Fast, Balanced, Deep), strong empty state, a few useful example prompts, clear loading and error states, streamed answer rendering, visible route/confidence/latency/source count, source citations, and an inspectable execution trace. Make citations open their matching source excerpt. Support Cmd/Ctrl+K to focus the query field. Enter should submit only with Cmd/Ctrl+Enter so multi-line prompts remain easy to edit.
2. **Pipeline metrics** - a genuinely separate page, not merely a collapsed widget inside the chat. Include retrieval quality (MRR, Recall@K, hit rate, nDCG, precision), confidence/evidence quality (citation accuracy, abstention accuracy), routing distribution (fast/balanced/deep), rerank/multi-query/parent-expansion/abstention rates, latency percentiles (P50/P90/P95/P99/mean/max), component/ablation comparison, and live system counts from `/health` and `/metrics`. Include clear time/data scope labels. The current repository README's 110-case adaptive benchmark reports MRR 0.7890, Recall@K 0.8556, Precision@K 0.5750, Hit Rate@K 84.4%, nDCG@K 0.7723, Abstention Accuracy 97.3%, Citation Accuracy 96.7%, full end-to-end P50 863.9 ms/P95 1504.4 ms/mean 801.8 ms, fast-path 3.6%, balanced 55.5%, deep 40.9%, rerank 87.3%, multi-query 28.2%, and abstention 20.9%. Distinguish end-to-end latency including generation from retrieval-only ablation latency in the low single-digit milliseconds. Never put these in a live KPI display unless they are explicitly labelled recorded benchmark results.
3. **Search explorer** - let users compare Hybrid, Dense, and BM25 retrieval and inspect ranked passages, scores, chunk identifiers, and source metadata before generation.
4. **Document library** - filter/search indexed documents, show index states, chunk counts and source type, and support upload if the API endpoint is available. Avoid pretending that upload succeeded when an API request fails.

## Interaction and quality requirements

- Implement actual interactions for navigation, mode controls, query submission, SSE parsing, copy answer, source tabs, search, document search/filter/upload, settings, API connection status and refresh.
- Build robust loading, empty, error, disconnected, and abstention states. Avoid false success messages.
- Add focus-visible states, keyboard support, accessible names, reduced-motion support, correct contrast, responsive layouts for desktop/tablet/mobile, and clear clickable affordances.
- Use compact charts only when they answer an analytical question. Use restrained chart color; show units; label source and scope.
- Keep reusable components, maintainable code and a README with setup instructions and API assumptions.
- The interface should look complete with seeded preview data but remain honest about connectivity and source of metrics.

## Desired output

A complete runnable frontend project, with source code and setup documentation, delivered as a ZIP. Favor polished, functional product execution over a static mockup.
