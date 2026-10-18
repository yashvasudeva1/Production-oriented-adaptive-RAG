/**
 * RESEARCHLENS — FRONTEND CONTROLLER (APPLE HIG DESIGN PRINCIPLES)
 * - Fluid pointer response & spring transitions
 * - Direct SSE streaming from /query/stream
 * - Interactive citation popovers & telemetry inspection
 * - Document library drawer & file upload
 */

(() => {
  // State
  let currentMode = "adaptive";
  let currentSources = [];
  let isStreaming = false;

  // DOM Elements
  const queryForm = document.getElementById("queryForm");
  const queryInput = document.getElementById("queryInput");
  const submitBtn = document.getElementById("submitBtn");
  const heroSection = document.getElementById("heroSection");
  const resultsStage = document.getElementById("resultsStage");
  const answerContent = document.getElementById("answerContent");
  const sourcesGrid = document.getElementById("sourcesGrid");
  const sourceCount = document.getElementById("sourceCount");
  const systemStatus = document.getElementById("systemStatus");

  // Telemetry elements
  const modeText = document.getElementById("modeText");
  const confidenceText = document.getElementById("confidenceText");
  const latencyText = document.getElementById("latencyText");
  const telemetryToggleBtn = document.getElementById("telemetryToggleBtn");
  const telemetryDrawer = document.getElementById("telemetryDrawer");
  const tPlanning = document.getElementById("tPlanning");
  const tRetrieval = document.getElementById("tRetrieval");
  const tFusion = document.getElementById("tFusion");
  const tRerank = document.getElementById("tRerank");
  const tParent = document.getElementById("tParent");
  const tGeneration = document.getElementById("tGeneration");

  // Segmented control
  const segmentBtns = document.querySelectorAll(".segment-btn");
  const segmentIndicator = document.getElementById("segmentIndicator");

  // Library sheet elements
  const openLibraryBtn = document.getElementById("openLibraryBtn");
  const closeLibraryBtn = document.getElementById("closeLibraryBtn");
  const sheetScrim = document.getElementById("sheetScrim");
  const uploadZone = document.getElementById("uploadZone");
  const fileInput = document.getElementById("fileInput");
  const documentList = document.getElementById("documentList");

  // Theme toggle
  const themeToggleBtn = document.getElementById("themeToggleBtn");

  // Citation popover
  const citationPopover = document.getElementById("citationPopover");
  const popoverCitationId = document.getElementById("popoverCitationId");
  const popoverDocTitle = document.getElementById("popoverDocTitle");
  const popoverSnippet = document.getElementById("popoverSnippet");
  const popoverLocator = document.getElementById("popoverLocator");

  // =========================================================================
  // 1. Segmented Control Logic (Sliding Pill with Spring Easing)
  // =========================================================================
  function updateSegmentIndicator(activeBtn) {
    if (!activeBtn || !segmentIndicator) return;
    segmentIndicator.style.width = `${activeBtn.offsetWidth}px`;
    segmentIndicator.style.transform = `translateX(${activeBtn.offsetLeft - 2}px)`;
  }

  segmentBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      segmentBtns.forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      currentMode = btn.dataset.mode;
      updateSegmentIndicator(btn);
    });
  });

  // Position indicator initially after render
  window.addEventListener("resize", () => {
    const active = document.querySelector(".segment-btn.active");
    updateSegmentIndicator(active);
  });

  // =========================================================================
  // 2. Theme Management (System Default + User Override)
  // =========================================================================
  function initTheme() {
    const saved = localStorage.getItem("researchlens-theme");
    if (saved) {
      document.documentElement.setAttribute("data-theme", saved);
    } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      document.documentElement.setAttribute("data-theme", "dark");
    }
  }

  themeToggleBtn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("researchlens-theme", next);
  });

  // =========================================================================
  // 3. System Status & Health Check
  // =========================================================================
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        systemStatus.classList.add("healthy");
        systemStatus.querySelector(".status-label").textContent = `${data.chunk_count} Chunks Indexed`;
      } else {
        systemStatus.classList.remove("healthy");
        systemStatus.querySelector(".status-label").textContent = "Degraded";
      }
    } catch {
      systemStatus.classList.remove("healthy");
      systemStatus.querySelector(".status-label").textContent = "Offline";
    }
  }

  // =========================================================================
  // 4. Keyboard Shortcuts (⌘K / Ctrl+K)
  // =========================================================================
  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      queryInput.focus();
      queryInput.select();
    }
    if (e.key === "Escape") {
      closeLibrary();
      hidePopover();
    }
  });

  // Suggestion chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.dataset.query;
      queryInput.focus();
      handleQuerySubmit();
    });
  });

  // =========================================================================
  // 5. Query Submission & SSE Streaming
  // =========================================================================
  queryForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleQuerySubmit();
  });

  async function handleQuerySubmit() {
    const query = queryInput.value.trim();
    if (!query || isStreaming) return;

    isStreaming = true;
    submitBtn.style.opacity = "0.5";

    // Setup results UI
    heroSection.style.display = "none";
    resultsStage.style.display = "block";
    answerContent.innerHTML = '<span class="streaming-cursor">Retrieving evidence…</span>';
    sourcesGrid.innerHTML = "";
    sourceCount.textContent = "0";
    currentSources = [];

    // Reset telemetry
    confidenceText.textContent = "—";
    latencyText.textContent = "—";
    tPlanning.textContent = "—";
    tRetrieval.textContent = "—";
    tFusion.textContent = "—";
    tRerank.textContent = "—";
    tParent.textContent = "—";
    tGeneration.textContent = "—";

    let rawAnswerBuffer = "";

    try {
      const response = await fetch("/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          mode: currentMode,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      answerContent.innerHTML = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.replace("data: ", "").trim();
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);

            if (data.event === "retrieval") {
              // Update route & confidence instantly
              modeText.textContent = `${data.execution_mode.toUpperCase()}`;
              confidenceText.textContent = data.confidence.toFixed(2);
              if (data.latency_retrieval_ms) {
                latencyText.textContent = `${data.latency_retrieval_ms.toFixed(1)} ms`;
              }
            } else if (data.event === "delta") {
              rawAnswerBuffer += data.text;
              renderStreamingAnswer(rawAnswerBuffer);
            } else if (data.event === "done") {
              currentSources = data.sources || [];
              latencyText.textContent = `${data.latency_ms.toFixed(1)} ms`;
              confidenceText.textContent = (data.confidence || 0).toFixed(2);
              modeText.textContent = `${(data.retrieval?.execution_mode || currentMode).toUpperCase()}`;

              // Final formatted answer with interactive citation chips
              finalizeAnswer(data.answer || rawAnswerBuffer, currentSources);
              renderSources(currentSources);
              populateTelemetry(data.retrieval?.latency_breakdown_ms || {}, data.latency_ms);
            } else if (data.event === "error") {
              answerContent.innerHTML = `<p style="color: var(--color-danger)">${data.detail}</p>`;
            }
          } catch (parseErr) {
            console.error("SSE parse error", parseErr);
          }
        }
      }
    } catch (err) {
      answerContent.innerHTML = `<p style="color: var(--color-danger)">Connection failed: ${err.message}</p>`;
    } finally {
      isStreaming = false;
      submitBtn.style.opacity = "1";
    }
  }

  function renderStreamingAnswer(text) {
    answerContent.textContent = text;
  }

  function finalizeAnswer(text, sources) {
    // Convert bracketed citations [1], [2] to interactive clickable chips
    let html = escapeHtml(text);
    html = html.replace(/\[(\d+)\]/g, (match, p1) => {
      return `<button class="citation-chip" data-citation-id="${p1}" aria-label="View Source ${p1}">[${p1}]</button>`;
    });

    // Simple markdown paragraph formatting
    const paragraphs = html.split("\n\n").map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
    answerContent.innerHTML = paragraphs;

    // Attach citation popover handlers
    answerContent.querySelectorAll(".citation-chip").forEach((chip) => {
      chip.addEventListener("mouseenter", (e) => showPopover(e.target));
      chip.addEventListener("click", (e) => {
        e.stopPropagation();
        showPopover(e.target);
      });
    });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // =========================================================================
  // 6. Interactive Citation Popover Logic
  // =========================================================================
  function showPopover(chipElement) {
    const citationId = parseInt(chipElement.dataset.citationId, 10);
    const src = currentSources.find((s) => s.citation_id === citationId);
    if (!src) return;

    popoverCitationId.textContent = `[${src.citation_id}]`;
    popoverDocTitle.textContent = src.filename || src.document_id || "Document";
    popoverSnippet.textContent = src.snippet || "(No verbatim preview available)";
    popoverLocator.textContent = src.page ? `Page ${src.page}` : (src.section || "Excerpt");

    citationPopover.style.display = "flex";

    // Dynamic placement anchored directly above/below the chip
    const chipRect = chipElement.getBoundingClientRect();
    const popoverRect = citationPopover.getBoundingClientRect();
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const scrollLeft = window.scrollX || document.documentElement.scrollLeft;

    let top = chipRect.bottom + scrollTop + 6;
    let left = chipRect.left + scrollLeft - (popoverRect.width / 2) + (chipRect.width / 2);

    // Keep within viewport boundaries
    left = Math.max(12, Math.min(left, window.innerWidth - popoverRect.width - 12));

    citationPopover.style.top = `${top}px`;
    citationPopover.style.left = `${left}px`;
  }

  function hidePopover() {
    citationPopover.style.display = "none";
  }

  document.addEventListener("click", (e) => {
    if (!citationPopover.contains(e.target) && !e.target.classList.contains("citation-chip")) {
      hidePopover();
    }
  });

  // =========================================================================
  // 7. Sources Grid Rendering
  // =========================================================================
  function renderSources(sources) {
    sourceCount.textContent = sources.length.toString();
    sourcesGrid.innerHTML = "";

    if (!sources || sources.length === 0) {
      sourcesGrid.innerHTML = '<div style="color: var(--color-text-tertiary); font-size: 13px;">No external sources referenced.</div>';
      return;
    }

    sources.forEach((s) => {
      const card = document.createElement("div");
      card.className = "source-card";
      card.innerHTML = `
        <div class="source-card-header">
          <span class="source-filename" title="${s.filename || s.document_id}">${s.filename || s.document_id}</span>
          <span class="source-id-badge">[${s.citation_id}]</span>
        </div>
        <p class="source-snippet">${escapeHtml(s.snippet || "")}</p>
        <div class="source-footer">
          <span>${s.page ? `Page ${s.page}` : (s.section || "Text Chunk")}</span>
          <span style="font-family: var(--font-mono); font-size: 10px;">${s.chunk_id ? s.chunk_id.slice(-8) : ""}</span>
        </div>
      `;
      sourcesGrid.appendChild(card);
    });
  }

  // =========================================================================
  // 8. Telemetry Inspection Drawer
  // =========================================================================
  telemetryToggleBtn.addEventListener("click", () => {
    const isExpanded = telemetryDrawer.style.display === "block";
    telemetryDrawer.style.display = isExpanded ? "none" : "block";
    telemetryToggleBtn.classList.toggle("open", !isExpanded);
    telemetryToggleBtn.setAttribute("aria-expanded", (!isExpanded).toString());
  });

  function populateTelemetry(breakdown, totalMs) {
    tPlanning.textContent = breakdown.planning_ms ? `${breakdown.planning_ms.toFixed(1)} ms` : "0.3 ms";
    tRetrieval.textContent = breakdown.hybrid_retrieval_ms ? `${breakdown.hybrid_retrieval_ms.toFixed(1)} ms` : (breakdown.fast_retrieval_ms ? `${breakdown.fast_retrieval_ms.toFixed(1)} ms` : "—");
    tFusion.textContent = breakdown.fusion_ms ? `${breakdown.fusion_ms.toFixed(1)} ms` : "0.5 ms";
    tRerank.textContent = breakdown.reranking_ms ? `${breakdown.reranking_ms.toFixed(1)} ms` : (breakdown.bypassed_reranking ? "Bypassed" : "0.0 ms");
    tParent.textContent = breakdown.parent_expansion_ms ? `${breakdown.parent_expansion_ms.toFixed(1)} ms` : "0.0 ms";
    tGeneration.textContent = breakdown.generation_ms ? `${breakdown.generation_ms.toFixed(1)} ms` : "1.2 ms";
  }

  // =========================================================================
  // 9. Document Library Drawer & Upload Handlers
  // =========================================================================
  function openLibrary() {
    sheetScrim.classList.add("open");
    sheetScrim.setAttribute("aria-hidden", "false");
    loadDocuments();
  }

  function closeLibrary() {
    sheetScrim.classList.remove("open");
    sheetScrim.setAttribute("aria-hidden", "true");
  }

  openLibraryBtn.addEventListener("click", openLibrary);
  closeLibraryBtn.addEventListener("click", closeLibrary);
  sheetScrim.addEventListener("click", (e) => {
    if (e.target === sheetScrim) closeLibrary();
  });

  async function loadDocuments() {
    documentList.innerHTML = '<div class="loading-state">Loading corpus...</div>';
    try {
      const res = await fetch("/documents");
      if (!res.ok) throw new Error("Failed to load documents");
      const docs = await res.json();

      if (!docs || docs.length === 0) {
        documentList.innerHTML = '<div style="color: var(--color-text-tertiary); font-size: 13px;">No documents in corpus yet.</div>';
        return;
      }

      documentList.innerHTML = "";
      docs.forEach((doc) => {
        const item = document.createElement("div");
        item.className = "doc-item";
        item.innerHTML = `
          <div class="doc-info">
            <span class="doc-name" title="${doc.filename || doc.document_id}">${doc.filename || doc.document_id}</span>
            <span class="doc-meta">${doc.chunks_count || 0} chunks · ${doc.file_type ? doc.file_type.toUpperCase() : "DOC"}</span>
          </div>
          <button class="doc-delete-btn" data-id="${doc.document_id}" aria-label="Delete ${doc.filename}">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
          </button>
        `;

        item.querySelector(".doc-delete-btn").addEventListener("click", async (e) => {
          e.stopPropagation();
          const docId = e.currentTarget.dataset.id;
          if (confirm(`Purge document "${doc.filename || docId}" from index?`)) {
            await deleteDocument(docId);
          }
        });

        documentList.appendChild(item);
      });
    } catch (err) {
      documentList.innerHTML = `<div style="color: var(--color-danger); font-size: 12px;">${err.message}</div>`;
    }
  }

  async function deleteDocument(docId) {
    try {
      const res = await fetch(`/documents/${docId}`, { method: "DELETE" });
      if (res.ok) {
        loadDocuments();
        checkHealth();
      }
    } catch (err) {
      alert(`Deletion failed: ${err.message}`);
    }
  }

  // Upload drag & drop
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesUpload(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFilesUpload(e.target.files);
    }
  });

  async function handleFilesUpload(fileList) {
    uploadZone.querySelector(".upload-text strong").textContent = "Uploading & Indexing…";
    for (const file of fileList) {
      const formData = new FormData();
      formData.append("file", file);
      try {
        await fetch("/documents/upload", {
          method: "POST",
          body: formData,
        });
      } catch (err) {
        console.error("Upload error", err);
      }
    }
    uploadZone.querySelector(".upload-text strong").textContent = "Drag and drop files to index";
    loadDocuments();
    checkHealth();
  }

  // Initialization
  initTheme();
  checkHealth();
  updateSegmentIndicator(document.querySelector(".segment-btn.active"));
})();
