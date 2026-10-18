# ResearchLens - Adaptive RAG frontend

A light, Apple-inspired React interface for the Production-oriented Adaptive RAG project. Built around the repository's FastAPI endpoints and terminology: hybrid dense + BM25 retrieval, adaptive routing, confidence-gated reranking, grounded answers, citations, and evidence inspection.

## Included

- **Query workspace** - retrieval mode selection, streaming answers, confidence/latency/source summary, source references, and execution trace.
- **Pipeline metrics** - a dedicated page for routing distribution, recorded benchmark quality, latency distribution, retrieval ablation comparison, and live operational index counts.
- **Search explorer** - inspect raw passages through `/search` using hybrid, dense, or BM25 retrieval.
- **Document library** - document list, filters, file upload integration point, and index status.
- **API setup** - configure a backend base URL without editing the source.
- Responsive desktop and mobile layout, keyboard shortcut `⌘K` / `Ctrl+K`, accessible labels, reduced-motion support.

## Start locally

Requirements: Node.js 20+ (recommended).

```bash
npm install
npm run dev
```

Vite prints the local URL. The UI defaults to **preview mode**. To use the live backend, click **API & setup** and enter your FastAPI URL, for example `http://localhost:8000`.

To create a production bundle:

```bash
npm run build
npm run preview
```

## API integration notes

The frontend uses these repository endpoints:

- `GET /health`
- `GET /metrics`
- `POST /query/stream` (Server-Sent Event chunks returned by a POST response stream)
- `POST /search`

CORS is configured broadly in the inspected FastAPI `src/api/app.py`, but local production deployment should use an origin allowlist.

Document listing/upload endpoints can differ between deployed versions. This frontend tries `GET /documents` and uploads to `POST /documents/upload`; if your backend uses different routes, adjust `refreshOverview()` and `uploadFiles()` in `src/main.jsx` to match the actual routes (the query, search, health, and metrics paths are based on the repository source). Unsupported document calls do not take down the rest of the UI.

If the app is deployed under the **same origin** as FastAPI, leave the API URL blank and optionally set `VITE_API_BASE` to an empty string at build time. With Vite dev server and a separate backend port, enter the backend origin in API settings. If the API blocks browser requests, configure CORS or set up a Vite proxy.

## Metrics honesty

The visible quality and latency charts are based on the repository's recorded 110-case adaptive benchmark and ablation data. They are explicitly labelled as recorded benchmark data. The live status panel only displays live values returned by `/health` and `/metrics`. It does not invent real-time traffic, quality scores, or request counts. End-to-end latency is kept distinct from retrieval-only latency because the former includes generation.

## Design direction

- Warm light-gray canvas, white working surfaces, restrained cobalt actions.
- Inter-like typography and compact system-style iconography.
- Fine borders, low shadows, generous alignment, functional interactions.
- No gradient wallpaper, glowing blobs, robot artwork, random charts on the home page, or decorative AI clichés.

## Repository

https://github.com/yashvasudeva1/Production-oriented-adaptive-RAG
