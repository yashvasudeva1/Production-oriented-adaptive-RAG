"""Hugging Face Spaces (Gradio SDK) entrypoint.

Mounts a minimal Gradio UI at '/' onto the FastAPI app while keeping
/health, /ready, /metrics, /query, /query/stream, /search, /demo, and /docs live.
"""
from __future__ import annotations

import os

# Override HF Spaces' GRADIO_SSR_MODE=True before importing spaces/gradio so Node SSR does not bind port 7860.
os.environ["GRADIO_SSR_MODE"] = "False"

try:
    import spaces  # type: ignore
except ImportError:
    spaces = None  # type: ignore

import uvicorn
from fastapi.routing import APIRoute

from src.api.app import app as fastapi_app, state

_gpu = spaces.GPU if spaces is not None else (lambda fn: fn)


def build_app():
    try:
        import gradio as gr
    except ImportError:
        return fastapi_app

    # Let Gradio serve '/' in HF Spaces while keeping /health, /metrics, /query, /demo, /docs active.
    fastapi_app.router.routes = [
        r for r in fastapi_app.router.routes
        if not (isinstance(r, APIRoute) and r.path == "/")
    ]

    @_gpu
    def run_rag(question: str, mode: str):
        q = (question or "").strip()
        if not q:
            return "Please enter a question.", "", ""
        res = state.orchestrator.query(query=q, mode=None if mode == "adaptive" else mode)
        meta = (
            f"Mode: {res.execution_mode} | Confidence: {res.confidence:.2f} | "
            f"Abstained: {res.abstained} | Latency: {res.latency_ms:.1f} ms"
        )
        sources_md = "\n".join(
            f"- **[{s.get('citation_id')}] {s.get('filename')}** "
            f"({s.get('source_locator') or s.get('chunk_id')}): {s.get('snippet', '')[:220]}"
            for s in res.sources
        ) or "_No sources cited._"
        return res.answer, meta, sources_md

    with gr.Blocks(title="Nexus Adaptive RAG") as demo:
        gr.Markdown(
            "# Nexus — Production-Oriented Adaptive RAG\n"
            "Hybrid retrieval (BM25 + Qdrant), RRF fusion, confidence-gated cross-encoder reranking, "
            "and grounded citations. API docs at `/docs`, SSE demo at `/demo`, health at `/health`."
        )
        with gr.Row():
            question_in = gr.Textbox(
                label="Question",
                placeholder="Ask a question across indexed documents...",
                lines=3,
            )
            mode_in = gr.Dropdown(
                choices=["adaptive", "fast", "balanced", "deep"],
                value="adaptive",
                label="Routing Mode",
            )
        ask_btn = gr.Button("Ask Nexus", variant="primary")
        meta_out = gr.Textbox(label="Routing & Telemetry", interactive=False)
        answer_out = gr.Markdown(label="Grounded Answer")
        sources_out = gr.Markdown(label="Cited Sources")

        ask_btn.click(
            fn=run_rag,
            inputs=[question_in, mode_in],
            outputs=[answer_out, meta_out, sources_out],
        )

    return gr.mount_gradio_app(fastapi_app, demo, path="/", ssr_mode=False)


app = build_app()

if __name__ == "__main__":
    if spaces is not None and os.getenv("SPACE_ID"):
        try:
            from spaces import zero  # type: ignore
            if hasattr(zero, "startup"):
                zero.startup()
        except Exception as exc:
            print(f"spaces.zero.startup note: {exc}", flush=True)

    port = int(
        os.getenv("GRADIO_SERVER_PORT")
        or os.getenv("PORT")
        or os.getenv("API_PORT")
        or "7860"
    )
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
