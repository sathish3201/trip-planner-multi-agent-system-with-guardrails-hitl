"use strict";

/* =========================================================
   APPLICATION STATE
   ========================================================= */

let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";
let waitingForApproval = false;
let activeAbortController = null;
let currentProgress = 0;

const AGENT_LABELS = {
  supervisor: "🧭 Supervisor",
  guardrail: "🛡️ Guardrails",
  guardrails: "🛡️ Guardrails",
  flight: "✈️ Flight Agent",
  flight_agent: "✈️ Flight Agent",
  hotel: "🏨 Hotel Agent",
  hotel_agent: "🏨 Hotel Agent",
  weather: "🌦️ Weather Agent",
  weather_agent: "🌦️ Weather Agent",
  budget: "💰 Budget Agent",
  budget_agent: "💰 Budget Agent",
  itinerary: "🗓️ Itinerary Agent",
  itinerary_agent: "🗓️ Itinerary Agent"
};

const STAGE_PROGRESS = {
  guardrail: 15,
  guardrails: 15,
  supervisor: 30,
  flight: 48,
  hotel: 62,
  weather: 74,
  budget: 86,
  itinerary: 100
};

function getEl(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeStageId(raw) {
  return String(raw ?? "")
    .toLowerCase()
    .trim()
    .replace(/\s+/g, "_")
    .replace(/_agent$/, "");
}

function formatAgentLabel(agent) {
  const normalized = normalizeStageId(agent);
  return (
    AGENT_LABELS[normalized] ||
    AGENT_LABELS[String(agent ?? "").toLowerCase().trim()] ||
    String(agent || "Agent").replace(/_/g, " ")
  );
}

function abortActiveStream() {
  if (activeAbortController) {
    activeAbortController.abort();
    activeAbortController = null;
  }
}


/* =========================================================
   PROGRESS BAR & PER-AGENT STATUS CARDS
   ========================================================= */

function ensureProgressBar() {
  const section = getEl("workflowSection");
  if (!section || getEl("agentProgressWrap")) return;

  const wrap = document.createElement("div");
  wrap.id = "agentProgressWrap";
  wrap.className = "progress-panel";
  wrap.innerHTML = `
    <div class="progress-meta">
      <span id="streamStatusText">Ready</span>
      <strong id="agentProgressPercent">0%</strong>
    </div>
    <div class="progress-track">
      <div id="agentProgressBar" class="progress-fill"></div>
    </div>
  `;

  const reasoning = getEl("supervisorReasoning");
  if (reasoning && reasoning.parentNode === section) {
    section.insertBefore(wrap, reasoning);
  } else {
    section.prepend(wrap);
  }
}

function setProgress(value, statusText) {
  ensureProgressBar();
  const clamped = Math.max(0, Math.min(100, Math.round(value)));
  currentProgress = Math.max(currentProgress, clamped);

  const bar = getEl("agentProgressBar");
  const pct = getEl("agentProgressPercent");
  const statusEl = getEl("streamStatusText");

  if (bar) bar.style.width = `${currentProgress}%`;
  if (pct) pct.textContent = `${currentProgress}%`;
  if (statusEl && statusText) statusEl.textContent = statusText;
}

function resetAgentProgress() {
  currentProgress = 0;
  ensureProgressBar();

  const chips = getEl("agentChips");
  if (chips) chips.replaceChildren();

  const bar = getEl("agentProgressBar");
  const pct = getEl("agentProgressPercent");
  if (bar) bar.style.width = "0%";
  if (pct) pct.textContent = "0%";
}

function upsertAgentCard(rawId, label, status = "running", detail = "") {
  const chipsContainer = getEl("agentChips");
  if (!chipsContainer) return;

  const section = getEl("workflowSection");
  if (section) section.classList.remove("hidden");

  const stageId = normalizeStageId(rawId || label);
  const cardId = `agent-card-${stageId}`;
  let card = document.getElementById(cardId);

  const normalizedStatus = String(status || "running").toLowerCase();
  const stateClass =
    normalizedStatus === "completed" || normalizedStatus === "done"
      ? "completed"
      : normalizedStatus === "failed" || normalizedStatus === "error"
      ? "failed"
      : "running";

  const statusLabel =
    stateClass === "completed"
      ? "Completed"
      : stateClass === "failed"
      ? "Failed"
      : "Running";

  if (!card) {
    card = document.createElement("div");
    card.id = cardId;
    card.className = `agent-chip ${stateClass}`;
    chipsContainer.appendChild(card);
  } else {
    card.className = `agent-chip ${stateClass}`;
  }

  card.innerHTML = `
    <div class="agent-card-top">
      <span>${escapeHtml(label || formatAgentLabel(stageId))}</span>
      <span class="agent-card-status">${statusLabel}</span>
    </div>
    ${
      detail
        ? `<div class="agent-card-detail">${escapeHtml(detail)}</div>`
        : ""
    }
  `;

  if (stateClass === "completed" && STAGE_PROGRESS[stageId] !== undefined) {
    setProgress(
      STAGE_PROGRESS[stageId],
      `${label || formatAgentLabel(stageId)} completed`
    );
  } else if (stateClass === "running") {
    setProgress(
      Math.min(92, currentProgress + 8),
      `${label || formatAgentLabel(stageId)}: ${detail || "Working..."}`
    );
  }
}


/* =========================================================
   PROMPTS, LOADING, ERRORS & MARKDOWN
   ========================================================= */

function setPrompt(text) {
  const input = getEl("userInput");
  if (!input) return;
  input.value = text;
  input.focus();
}

function setLoading(isLoading, mode = "draft") {
  const sendBtn = getEl("sendBtn");
  const btnText = getEl("btnText");
  const btnLoader = getEl("btnLoader");
  const approveBtn = getEl("approveBtn");
  const reviseBtn = getEl("reviseBtn");

  if (sendBtn) sendBtn.disabled = isLoading || waitingForApproval;
  if (approveBtn) approveBtn.disabled = isLoading;
  if (reviseBtn) reviseBtn.disabled = isLoading;

  if (mode === "draft" && btnText && btnLoader) {
    btnText.textContent = isLoading ? "Crafting..." : "Craft Itinerary";
    btnText.classList.toggle("hidden", isLoading);
    btnLoader.classList.toggle("hidden", !isLoading);
  }
}

function showError(message) {
  const errorBox = getEl("errorBox");
  if (!errorBox) return;

  errorBox.textContent = message || "Something went wrong.";
  errorBox.classList.remove("hidden");
  errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideError() {
  const errorBox = getEl("errorBox");
  if (!errorBox) return;

  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function renderMarkdown(element, markdown) {
  if (!element) return;

  const source = String(markdown ?? "");
  if (typeof marked !== "undefined" && typeof marked.parse === "function") {
    const html = marked.parse(source);
    element.innerHTML = typeof DOMPurify !== "undefined" ? DOMPurify.sanitize(html) : html;
  } else {
    element.textContent = source;
  }
}


/* =========================================================
   WORKFLOW, RESULT & HUMAN APPROVAL UI
   ========================================================= */

function showWorkflow(data = {}) {
  const section = getEl("workflowSection");
  const reasoning = getEl("supervisorReasoning");
  const guardrailBadge = getEl("guardrailBadge");

  ensureProgressBar();

  if (reasoning && data.supervisor_reasoning) {
    reasoning.textContent = data.supervisor_reasoning;
  }

  if (Array.isArray(data.selected_agents)) {
    data.selected_agents.forEach((agent) => {
      const id = normalizeStageId(agent);
      if (!document.getElementById(`agent-card-${id}`)) {
        upsertAgentCard(agent, formatAgentLabel(agent), "running", "Queued by supervisor");
      }
    });
  }

  if (guardrailBadge && typeof data.guardrail_allowed === "boolean") {
    if (data.guardrail_allowed) {
      guardrailBadge.textContent = "Guardrails Verified";
      guardrailBadge.classList.remove("blocked");
    } else {
      guardrailBadge.textContent = "Guardrail Blocked";
      guardrailBadge.classList.add("blocked");
    }
  }

  if (section) {
    section.classList.remove("hidden");
  }
}

function updateStreamProgress(statusMessage) {
  setProgress(currentProgress, statusMessage);
  const reasoning = getEl("supervisorReasoning");
  const section = getEl("workflowSection");

  if (reasoning && statusMessage && !reasoning.textContent) {
    reasoning.textContent = statusMessage;
  }
  if (section) {
    section.classList.remove("hidden");
  }
}

function showResult(answer, threadId, isDraft = false) {
  latestAnswerMarkdown = String(answer ?? "");

  const resultSection = getEl("resultSection");
  const resultBox = getEl("resultBox");
  const threadInfo = getEl("threadInfo");
  const resultTitle = getEl("resultTitle");

  renderMarkdown(resultBox, latestAnswerMarkdown);

  if (threadInfo && threadId) {
    threadInfo.textContent = `Thread ID: ${threadId}`;
  }

  if (resultTitle) {
    resultTitle.textContent = isDraft ? "Draft Travel Plan" : "Your Custom Itinerary";
  }

  if (resultSection) {
    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function showApproval(data = {}) {
  waitingForApproval = true;
  const section = getEl("approvalSection");
  const approvalRequest = getEl("approvalRequest");
  const sendBtn = getEl("sendBtn");

  if (approvalRequest) {
    approvalRequest.textContent =
      data.approval_request ||
      "Approve this draft or request custom adjustments before finalizing.";
  }

  if (sendBtn) sendBtn.disabled = true;
  if (section) section.classList.remove("hidden");
}

function hideApproval() {
  waitingForApproval = false;
  const section = getEl("approvalSection");
  const feedback = getEl("approvalFeedback");
  const sendBtn = getEl("sendBtn");

  if (section) section.classList.add("hidden");
  if (feedback) feedback.value = "";
  if (sendBtn) sendBtn.disabled = false;
}


/* =========================================================
   SSE STREAMING HANDLER (POST /api/run)
   ========================================================= */

function handleStreamEvent(event) {
  if (!event || typeof event !== "object") return;

  switch (event.type) {
    case "run_started": {
      if (event.thread_id) {
        currentThreadId = event.thread_id;
        localStorage.setItem("travel_thread_id", currentThreadId);
      }
      setProgress(8, "Multi-agent workflow started...");
      break;
    }

    case "stage": {
      const stageId = event.id || event.agent || event.label;
      const label = event.label || formatAgentLabel(stageId);
      upsertAgentCard(
        stageId,
        label,
        event.status || "completed",
        event.detail || ""
      );
      showWorkflow(event);
      break;
    }

    case "agent_event": {
      const agentKey = event.id || event.agent || event.event || "agent";
      const label = formatAgentLabel(agentKey);
      const detail = event.message || event.data?.message || "Processing...";
      upsertAgentCard(agentKey, label, event.status || "running", detail);
      break;
    }

    case "status": {
      updateStreamProgress(event.content || "Working...");
      break;
    }

    case "workflow":
    case "state_update": {
      showWorkflow(event.data || event);
      break;
    }

    case "token": {
      latestAnswerMarkdown += String(event.content ?? "");
      renderMarkdown(getEl("resultBox"), latestAnswerMarkdown);
      getEl("resultSection")?.classList.remove("hidden");
      break;
    }

    case "final":
    case "complete": {
      const payload = event.result || event.data || event;

      if (payload.thread_id) {
        currentThreadId = payload.thread_id;
        localStorage.setItem("travel_thread_id", currentThreadId);
      }

      showWorkflow(payload);
      (payload.selected_agents || []).forEach((agent) => {
        upsertAgentCard(agent, formatAgentLabel(agent), "completed", "Done");
      });

      const planMarkdown = payload.itinerary || payload.answer || latestAnswerMarkdown;
      if (payload.requires_approval) {
        setProgress(95, "Draft ready — waiting for your approval");
        showResult(planMarkdown, payload.thread_id, true);
        showApproval(payload);
      } else {
        setProgress(100, "Itinerary ready");
        hideApproval();
        showResult(planMarkdown, payload.thread_id, false);
      }

      setLoading(false, "draft");
      break;
    }

    case "done": {
      setLoading(false, "draft");
      break;
    }

    case "error": {
      throw new Error(event.message || event.error || "Streaming error encountered.");
    }

    default: {
      console.debug("Unhandled SSE event:", event);
    }
  }
}

function parseSSEBuffer(buffer) {
  const blocks = buffer.split("\n\n");
  const remainder = blocks.pop() || "";

  for (const block of blocks) {
    const dataLines = block
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart());

    if (!dataLines.length) continue;

    const payload = JSON.parse(dataLines.join("\n"));
    handleStreamEvent(payload);
  }

  return remainder;
}

async function sendMessage() {
  hideError();
  abortActiveStream();

  if (waitingForApproval) {
    showError("Please approve or revise the current draft before starting another plan.");
    return;
  }

  const input = getEl("userInput");
  const message = input ? input.value.trim() : "";

  if (!message) {
    showError("Please enter your travel request first.");
    input?.focus();
    return;
  }

  latestAnswerMarkdown = "";
  getEl("resultSection")?.classList.add("hidden");
  renderMarkdown(getEl("resultBox"), "");
  resetAgentProgress();
  setProgress(10, "Connecting to travel workflow...");
  upsertAgentCard("supervisor", "🧭 Supervisor", "running", "Routing your request...");
  setLoading(true, "draft");

  activeAbortController = new AbortController();

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream"
      },
      body: JSON.stringify({
        query: message,
        thread_id: currentThreadId
      }),
      signal: activeAbortController.signal
    });

    if (!response.ok) {
      throw new Error((await response.text()) || "Could not start travel workflow.");
    }

    if (!response.body) {
      throw new Error("Streaming is not supported in this browser.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      buffer = parseSSEBuffer(buffer);
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      parseSSEBuffer(`${buffer.replace(/\r\n/g, "\n")}\n\n`);
    }
  } catch (error) {
    if (error.name === "AbortError") {
      setProgress(currentProgress, "Stopped");
    } else {
      console.error("Travel planning error:", error);
      showError(error.message || "Unable to generate your travel plan.");
      setProgress(currentProgress, "Failed");
    }
  } finally {
    setLoading(false, "draft");
    activeAbortController = null;
  }
}

async function submitApproval(approved) {
  hideError();

  if (!currentThreadId || !waitingForApproval) {
    showError("There is no draft waiting for approval.");
    return;
  }

  const feedbackInput = getEl("approvalFeedback");
  const feedback = feedbackInput ? feedbackInput.value.trim() : "";

  if (!approved && !feedback) {
    showError("Please enter revision feedback before requesting changes.");
    feedbackInput?.focus();
    return;
  }

  setLoading(true, "approval");
  setProgress(96, approved ? "Finalizing approved plan..." : "Applying revision feedback...");

  try {
    const response = await fetch("/api/approval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thread_id: currentThreadId,
        approved,
        feedback
      })
    });

    const data = await response.json();
    if (!response.ok || data.success === false) {
      throw new Error(data.error || data.detail || "Could not resume travel workflow.");
    }

    if (data.thread_id) {
      currentThreadId = data.thread_id;
      localStorage.setItem("travel_thread_id", currentThreadId);
    }

    showWorkflow(data);
    const nextPlan = data.itinerary || data.answer || latestAnswerMarkdown;

    if (data.requires_approval) {
      showResult(nextPlan, data.thread_id, true);
      showApproval(data);
    } else {
      setProgress(100, "Completed");
      hideApproval();
      showResult(nextPlan, data.thread_id, false);
    }
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false, "approval");
  }
}


/* =========================================================
   COPY, PDF EXPORT & KEYBOARD SHORTCUTS
   ========================================================= */

function copyResult() {
  const resultBox = getEl("resultBox");
  const text = latestAnswerMarkdown || resultBox?.innerText || "";
  if (!text) return;

  navigator.clipboard.writeText(text).then(() => {
    const copyBtn = document.querySelector(".copy-btn");
    if (!copyBtn) return;

    const oldText = copyBtn.textContent;
    copyBtn.textContent = "Copied!";
    setTimeout(() => {
      copyBtn.textContent = oldText;
    }, 1400);
  });
}

function downloadPDF() {
  const pdfContent = getEl("pdfContent");
  const pdfTitle = document.querySelector(".pdf-title");

  if (!latestAnswerMarkdown || !pdfContent || typeof html2pdf === "undefined") {
    showError("No travel plan available to download.");
    return;
  }

  const downloadBtn = document.querySelector(".download-btn");
  const oldText = downloadBtn ? downloadBtn.textContent : "Export PDF";
  if (downloadBtn) {
    downloadBtn.textContent = "Preparing PDF...";
    downloadBtn.disabled = true;
  }
  if (pdfTitle) pdfTitle.style.display = "block";

  html2pdf()
    .set({
      margin: 0.5,
      filename: "VoyageCraft-Travel-Plan.pdf",
      image: { type: "jpeg", quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
      jsPDF: { unit: "in", format: "a4", orientation: "portrait" },
      pagebreak: { mode: ["avoid-all", "css", "legacy"] }
    })
    .from(pdfContent)
    .save()
    .finally(() => {
      if (pdfTitle) pdfTitle.style.display = "";
      if (downloadBtn) {
        downloadBtn.textContent = oldText;
        downloadBtn.disabled = false;
      }
    });
}

document.addEventListener("DOMContentLoaded", () => {
  ensureProgressBar();
  if (currentThreadId) {
    const threadInfo = getEl("threadInfo");
    if (threadInfo) threadInfo.textContent = `Thread ID: ${currentThreadId}`;
  }
});

document.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") {
    event.preventDefault();
    sendMessage();
  }
});