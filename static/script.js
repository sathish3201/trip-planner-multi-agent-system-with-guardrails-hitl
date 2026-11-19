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
  guardrail_blocked: "🛑 Guardrail Block",
  flight: "✈️ Flight Agent",
  flight_agent: "✈️ Flight Agent",
  hotel: "🏨 Hotel Agent",
  hotel_agent: "🏨 Hotel Agent",
  weather: "🌦️ Weather Agent",
  weather_agent: "🌦️ Weather Agent",
  budget: "💰 Budget Agent",
  budget_agent: "💰 Budget Agent",
  itinerary: "🗓️ Itinerary Agent",
  itinerary_agent: "🗓️ Itinerary Agent",
  approval: "🧑‍⚖️ Human Approval",
  final: "✅ Final Agent",
  final_agent: "✅ Final Agent"
};

const STAGE_PROGRESS = {
  guardrail: 15,
  guardrails: 15,
  supervisor: 30,
  flight: 48,
  hotel: 62,
  weather: 74,
  budget: 86,
  itinerary: 95,
  approval: 95,
  final: 100
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
   PROMPTS, LOADING, ERRORS & MARKDOWN NORMALIZATION
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

function cleanSpecialCharacters(raw) {
  return String(raw ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    // Strip control & invisible/zero-width characters except \n and \t
    .replace(/[\u0000-\u0008\u000B-\u001F\u007F-\u009F\u200B-\u200F\uFEFF]/g, "")
    // Normalize non-breaking space and weird dashes/quotes to standard equivalents
    .replace(/\u00A0/g, " ")
    .replace(/[\u2010-\u2015]/g, "-")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/…/g, "...")
    // Remove stray replacement characters or decorative box/symbol noise
    .replace(/\uFFFD/g, "")
    .replace(/[□■▪▫◆◇★☆✦✧●○◎]/g, "")
    // Collapse multiple spaces (keeping indentation/newlines intact)
    .replace(/[ \t]{2,}/g, " ");
}

function isPureDividerToken(str) {
  const s = String(str ?? "").trim();
  return /^\|?[\s\-:]+\|?$/.test(s) && s.includes("-");
}

function splitRowIntoCells(line) {
  let s = String(line ?? "").trim();
  if (!s.includes("|")) return [s];

  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);

  return s.split("|").map((c) => c.trim());
}

function formatTableRow(cells, expectedCols) {
  const cleaned = [...cells];

  if (cleaned.length > expectedCols && cleaned[0] === "") {
    cleaned.shift();
  }
  while (cleaned.length > expectedCols && cleaned[cleaned.length - 1] === "") {
    cleaned.pop();
  }

  if (cleaned.length < expectedCols) {
    while (cleaned.length < expectedCols) {
      cleaned.push("");
    }
  } else if (cleaned.length > expectedCols) {
    const head = cleaned.slice(0, expectedCols - 1);
    const tail = cleaned.slice(expectedCols - 1).filter(Boolean).join(" · ");
    return `| ${[...head, tail].join(" | ")} |`;
  }

  return `| ${cleaned.join(" | ")} |`;
}

function buildDividerRow(rawDivider, colCount) {
  const parts = String(rawDivider ?? "")
    .replace(/^\|+|\|+$/g, "")
    .split("|")
    .map((p) => p.trim())
    .filter((p) => p.includes("-"));

  const out = [];
  for (let i = 0; i < colCount; i += 1) {
    const p = parts[i] || "---";
    const left = p.startsWith(":");
    const right = p.endsWith(":");
    if (left && right) out.push(":---:");
    else if (left) out.push(":---");
    else if (right) out.push("---:");
    else out.push("---");
  }
  return `| ${out.join(" | ")} |`;
}

function repairLineStructure(raw) {
  let text = cleanSpecialCharacters(raw);

  // Break inline headings or blockquotes onto separate lines
  text = text.replace(/([^\n])\s+(#{1,6}\s+)/g, "$1\n\n$2");
  text = text.replace(/([^\n])\s+(>\s+)/g, "$1\n\n$2");

  // Put standalone bold section titles on their own paragraph
  text = text.replace(
    /\s*(\*\*(?:Missing Trip Details[^*]*|Day \d+\s*-\s*[^*]+|Budget Summary[^*]*|Quick Verification Before Sending)\*\*)\s*/g,
    "\n\n$1\n\n"
  );

  // Move inline "- **Label:**" or timed bullets onto their own line
  text = text.replace(/\s+-\s+(\*\*[^*\n]+:\*\*)/g, "\n- $1");
  text = text.replace(
    /\s+-\s+((?:Departure|Travel|Budget|Morning|Mid-day|Mid-afternoon|Early afternoon|Afternoon|Late afternoon|Evening|Night|Hotel|Flight|Transfer)[^:\n]*:)/g,
    "\n- **$1**"
  );

  // Ensure a blank line between a bold section header and its first bullet item
  text = text.replace(/(\*\*[^*\n]+\*\*)\n(- )/g, "$1\n\n$2");

  // Put inline numbered verification questions on separate lines
  text = text.replace(/([^\n])\s+(\d+\.\s+[A-Z])/g, "$1\n$2");

  // Break inline horizontal rules onto their own line
  text = text.replace(/([^\n-])\s+---\s+([^\n-])/g, "$1\n\n---\n\n$2");

  // Break adjacent table rows glued onto a single line ("| ... | | ... |" -> "| ... |\n| ... |")
  text = text.replace(/\|\s+\|(?=\s*[\*|a-zA-Z0-9$₹#])/g, "|\n|");

  // Separate heading text from an inline table header on the same line
  text = text.replace(
    /^(#{1,6}\s+[^|\n]+?)\s+(\|[^|\n]+\|[^|\n]+\|[^\n]*)$/gm,
    "$1\n\n$2"
  );

  return text;
}

function normalizeMarkdownTables(text) {
  const lines = text.split("\n");
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const cur = lines[i].trim();

    if (cur.includes("|") && !isPureDividerToken(cur) && !/^#{1,6}\s/.test(cur)) {
      let j = i + 1;
      const dividerFragments = [];

      while (j < lines.length) {
        const peek = lines[j].trim();
        if (!peek || peek === "---") {
          j += 1;
          continue;
        }
        if (isPureDividerToken(peek)) {
          dividerFragments.push(peek);
          j += 1;
          continue;
        }
        break;
      }

      if (dividerFragments.length > 0) {
        const headerCells = splitRowIntoCells(cur).filter(Boolean);
        const colCount = Math.max(2, headerCells.length);

        out.push("");
        out.push(formatTableRow(headerCells, colCount));
        out.push(buildDividerRow(dividerFragments.join("|"), colCount));

        i = j;
        while (i < lines.length) {
          const rowLine = lines[i].trim();
          if (!rowLine) {
            i += 1;
            continue;
          }
          if (rowLine === "---" || isPureDividerToken(rowLine)) {
            i += 1;
            continue;
          }
          if (/^#{1,6}\s/.test(rowLine) || /^>\s/.test(rowLine)) {
            break;
          }
          if (rowLine.includes("|")) {
            out.push(formatTableRow(splitRowIntoCells(rowLine), colCount));
            i += 1;
            continue;
          }
          break;
        }
        out.push("");
        continue;
      }
    }

    out.push(lines[i]);
    i += 1;
  }

  return out.join("\n");
}

function normalizeMarkdown(markdown) {
  const structured = repairLineStructure(markdown);
  const withTables = normalizeMarkdownTables(structured);

  return withTables
    .replace(/([^\n])\n(#{1,6}\s)/g, "$1\n\n$2")
    .replace(/([^\n])\n(>\s?)/g, "$1\n\n$2")
    .replace(/^(#{1,6}\s+)\*\*(.+?)\*\*\s*$/gm, "$1$2")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function renderMarkdown(element, markdown) {
  if (!element) return;

  const source = normalizeMarkdown(markdown);
  if (typeof marked !== "undefined" && typeof marked.parse === "function") {
    const html = marked.parse(source, { gfm: true, breaks: false });
    element.innerHTML =
      typeof DOMPurify !== "undefined" ? DOMPurify.sanitize(html) : html;
  } else {
    element.textContent = source;
  }
}


/* =========================================================
   TELEMETRY PANEL (TOKENS, LLM CALLS, LATENCY, COST, ACCURACY)
   ========================================================= */

function renderTelemetry(data = {}) {
  const panel = getEl("telemetrySection");
  if (!panel) return;

  const totalTokens = Number(data.token_usage ?? 0);
  const tokenLimit = Number(data.token_limit ?? 0);
  const remainingTokens = Number(
    data.remaining_tokens ?? Math.max(0, tokenLimit - totalTokens)
  );
  const llmCalls = Number(data.llm_calls ?? 0);

  const latencyMs = Number(data.overall_latency_ms ?? data.latency_ms ?? 0);
  const latencySec = (latencyMs / 1000).toFixed(2);
  const completedTime = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });

  const costUsd = Number(data.total_cost_usd ?? 0);
  const costInr = Number(data.total_cost_inr ?? costUsd * 84);
  const accuracy = Number(data.overall_accuracy ?? data.accuracy_score ?? 100);

  const agentMetrics =
    data.agent_metrics && typeof data.agent_metrics === "object"
      ? data.agent_metrics
      : {};

  let inputSum = 0;
  let outputSum = 0;
  Object.values(agentMetrics).forEach((m) => {
    inputSum += Number(m?.input_tokens ?? 0);
    outputSum += Number(m?.output_tokens ?? 0);
  });

  const metricTokens = getEl("metricTokens");
  const metricTokenBreakdown = getEl("metricTokenBreakdown");
  const metricLlmCalls = getEl("metricLlmCalls");
  const metricBudgetRemaining = getEl("metricBudgetRemaining");
  const metricLatency = getEl("metricLatency");
  const metricRunTime = getEl("metricRunTime");
  const metricCost = getEl("metricCost");
  const metricCostInr = getEl("metricCostInr");
  const metricAccuracy = getEl("metricAccuracy");
  const metricAccuracyDetail = getEl("metricAccuracyDetail");

  if (metricTokens) metricTokens.textContent = totalTokens.toLocaleString();
  if (metricTokenBreakdown) {
    metricTokenBreakdown.textContent = data.cached
      ? "Served from cache"
      : inputSum || outputSum
      ? `Input: ${inputSum.toLocaleString()} · Output: ${outputSum.toLocaleString()}`
      : tokenLimit
      ? `Limit: ${tokenLimit.toLocaleString()}`
      : "Total tokens consumed";
  }

  if (metricLlmCalls) {
    metricLlmCalls.textContent = data.cached ? "0 (Cached)" : String(llmCalls);
  }
  if (metricBudgetRemaining) {
    metricBudgetRemaining.textContent = data.cached
      ? "Cache Hit · Saved compute"
      : tokenLimit
      ? `${remainingTokens.toLocaleString()} tokens left`
      : "Model invocations";
  }

  if (metricLatency) metricLatency.textContent = `${latencySec}s`;
  if (metricRunTime) {
    metricRunTime.textContent = `${Math.round(latencyMs).toLocaleString()} ms · ${completedTime}`;
  }

  if (metricCost) {
    metricCost.textContent = data.cached ? "$0.000000" : `$${costUsd.toFixed(6)}`;
  }
  if (metricCostInr) {
    metricCostInr.textContent = data.cached ? "₹0.0000" : `₹${costInr.toFixed(4)}`;
  }

  if (metricAccuracy) metricAccuracy.textContent = `${accuracy.toFixed(1)}%`;
  if (metricAccuracyDetail) {
    metricAccuracyDetail.textContent =
      data.guardrail_allowed === false
        ? "Blocked by guardrails"
        : `${Object.keys(agentMetrics).length || (data.selected_agents || []).length} stages evaluated`;
  }

  const listEl = getEl("agentTelemetryList");
  if (listEl) {
    const entries = Object.entries(agentMetrics);
    if (!entries.length) {
      listEl.replaceChildren();
    } else {
      listEl.innerHTML = entries
        .map(([agentName, m]) => {
          const label = escapeHtml(formatAgentLabel(agentName));
          const aTokens = Number(m?.tokens ?? 0).toLocaleString();
          const aLat = Number(
            m?.latency_s ?? (m?.latency_ms ?? 0) / 1000
          ).toFixed(2);
          const aCostUsd = Number(m?.cost_usd ?? 0).toFixed(6);
          const aAcc = Number(m?.accuracy ?? 100).toFixed(1);
          return `
            <div class="agent-metric-item">
              <div class="agent-metric-title">${label}</div>
              <div>Tokens: <strong>${aTokens}</strong> · Accuracy: <strong>${aAcc}%</strong></div>
              <div>Latency: <strong>${aLat}s</strong> · Cost: <strong>$${aCostUsd}</strong></div>
            </div>
          `;
        })
        .join("");
    }
  }

  panel.classList.remove("hidden");
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
        upsertAgentCard(
          agent,
          formatAgentLabel(agent),
          "running",
          "Queued by supervisor"
        );
      }
    });
  }

  if (guardrailBadge && typeof data.guardrail_allowed === "boolean") {
    if (data.guardrail_allowed) {
      guardrailBadge.textContent = data.cached
        ? "Cache Hit · Verified"
        : "Guardrails Verified";
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
    resultTitle.textContent = isDraft
      ? "Draft Travel Plan"
      : "Your Custom Itinerary";
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
  if (section) {
    section.classList.remove("hidden");
    section.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
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
      renderTelemetry(payload);

      if (payload.cached) {
        upsertAgentCard(
          "supervisor",
          "🧭 Supervisor",
          "completed",
          "Cache hit — reused verified itinerary"
        );
      }

      (payload.selected_agents || []).forEach((agent) => {
        upsertAgentCard(
          agent,
          formatAgentLabel(agent),
          "completed",
          payload.cached ? "Cached" : "Done"
        );
      });

      const planMarkdown =
        payload.itinerary || payload.answer || latestAnswerMarkdown;

      if (payload.requires_approval) {
        upsertAgentCard(
          "itinerary",
          "🗓️ Itinerary Agent",
          "completed",
          "Draft ready"
        );
        upsertAgentCard(
          "approval",
          "🧑‍⚖️ Human Approval",
          "running",
          "Waiting for your review"
        );
        setProgress(95, "Draft ready — waiting for your approval");
        showResult(planMarkdown, payload.thread_id, true);
        showApproval(payload);
      } else {
        setProgress(
          100,
          payload.cached ? "Itinerary loaded from cache" : "Itinerary ready"
        );
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
      throw new Error(
        event.message || event.error || "Streaming error encountered."
      );
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
    showError(
      "Please approve or revise the current draft before starting another plan."
    );
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
  getEl("telemetrySection")?.classList.add("hidden");
  renderMarkdown(getEl("resultBox"), "");
  resetAgentProgress();
  setProgress(10, "Connecting to travel workflow...");
  upsertAgentCard(
    "supervisor",
    "🧭 Supervisor",
    "running",
    "Routing your request..."
  );
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
      throw new Error(
        (await response.text()) || "Could not start travel workflow."
      );
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

      buffer += decoder
        .decode(value, { stream: true })
        .replace(/\r\n/g, "\n");
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
  setProgress(
    96,
    approved ? "Finalizing approved plan..." : "Applying revision feedback..."
  );

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
      throw new Error(
        data.error || data.detail || "Could not resume travel workflow."
      );
    }

    if (data.thread_id) {
      currentThreadId = data.thread_id;
      localStorage.setItem("travel_thread_id", currentThreadId);
    }

    showWorkflow(data);
    renderTelemetry(data);
    const nextPlan = data.itinerary || data.answer || latestAnswerMarkdown;

    if (data.requires_approval) {
      upsertAgentCard(
        "approval",
        "🧑‍⚖️ Human Approval",
        "running",
        "Waiting for additional review"
      );
      showResult(nextPlan, data.thread_id, true);
      showApproval(data);
    } else {
      upsertAgentCard(
        "approval",
        "🧑‍⚖️ Human Approval",
        "completed",
        approved ? "Approved" : "Revised"
      );
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
  const text =
    normalizeMarkdown(latestAnswerMarkdown) || resultBox?.innerText || "";
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

function createPdfExportContainer(sourceElement, markdownText) {
  const wrapper = document.createElement("div");
  wrapper.className = "pdf-export-root";
  wrapper.style.cssText = [
    "position: fixed",
    "left: -9999px",
    "top: 0",
    "width: 794px",
    "padding: 24px",
    "background: #ffffff",
    "color: #111827",
    "font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
    "font-size: 13px",
    "line-height: 1.55",
    "z-index: 10"
  ].join("; ");

  const titleEl = document.querySelector(".pdf-title");
  if (titleEl) {
    const titleClone = titleEl.cloneNode(true);
    titleClone.style.display = "block";
    titleClone.style.color = "#111827";
    wrapper.appendChild(titleClone);
  }

  const body = document.createElement("div");
  if (markdownText) {
    renderMarkdown(body, markdownText);
  } else if (sourceElement) {
    body.innerHTML = sourceElement.innerHTML;
  }

  body.querySelectorAll("h1, h2, h3, h4, p, li, blockquote").forEach((el) => {
    el.style.color = "#111827";
  });
  body.querySelectorAll("blockquote").forEach((bq) => {
    bq.style.borderLeft = "3px solid #6366f1";
    bq.style.paddingLeft = "12px";
    bq.style.margin = "10px 0";
    bq.style.background = "#f8fafc";
  });
  body.querySelectorAll("table").forEach((tbl) => {
    tbl.style.width = "100%";
    tbl.style.borderCollapse = "collapse";
    tbl.style.margin = "12px 0";
  });
  body.querySelectorAll("th").forEach((th) => {
    th.style.border = "1px solid #cbd5e1";
    th.style.padding = "7px 8px";
    th.style.textAlign = "left";
    th.style.background = "#f1f5f9";
    th.style.color = "#0f172a";
    th.style.fontWeight = "600";
  });
  body.querySelectorAll("td").forEach((td) => {
    td.style.border = "1px solid #cbd5e1";
    td.style.padding = "6px 8px";
    td.style.textAlign = "left";
    td.style.color = "#111827";
    td.style.wordBreak = "break-word";
  });

  wrapper.appendChild(body);
  document.body.appendChild(wrapper);
  return wrapper;
}

async function downloadPDF() {
  const resultBox = getEl("resultBox");
  const pdfContent = getEl("pdfContent");
  const rawContent =
    latestAnswerMarkdown || (resultBox && resultBox.innerText.trim());

  if (!rawContent || typeof html2pdf === "undefined") {
    showError("No travel plan available to download or PDF library not loaded.");
    return;
  }

  const downloadBtn = document.querySelector(".download-btn");
  const oldText = downloadBtn ? downloadBtn.textContent : "Export PDF";
  if (downloadBtn) {
    downloadBtn.textContent = "Preparing PDF...";
    downloadBtn.disabled = true;
  }

  const exportRoot = createPdfExportContainer(resultBox, rawContent);
  if (pdfContent) {
    pdfContent.innerHTML = exportRoot.innerHTML;
  }

  try {
    await html2pdf()
      .set({
        margin: [0.5, 0.5, 0.5, 0.5],
        filename: "VoyageCraft-Travel-Plan.pdf",
        image: { type: "jpeg", quality: 0.98 },
        html2canvas: {
          scale: 2,
          useCORS: true,
          backgroundColor: "#ffffff",
          windowWidth: exportRoot.scrollWidth
        },
        jsPDF: { unit: "in", format: "a4", orientation: "portrait" },
        pagebreak: { mode: ["css", "legacy"] }
      })
      .from(exportRoot)
      .save();
  } catch (error) {
    console.error("PDF export error:", error);
    showError(error.message || "Failed to generate PDF.");
  } finally {
    exportRoot.remove();
    if (pdfContent) {
      pdfContent.replaceChildren();
    }
    if (downloadBtn) {
      downloadBtn.textContent = oldText;
      downloadBtn.disabled = false;
    }
  }
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