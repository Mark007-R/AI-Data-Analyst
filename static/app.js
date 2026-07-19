/* DataAI frontend: upload -> processing -> dashboard + chat */
"use strict";

let datasetId = null;
let statusMisses = 0;
const chatHistory = [];
const echartsInstances = [];

const $ = (id) => document.getElementById(id);

// Render untrusted markdown (dataset values / LLM output) safely: marked does NOT
// sanitize, so every marked output is passed through DOMPurify before hitting innerHTML.
const safeMd = (s) => DOMPurify.sanitize(marked.parse(s || ""));
const safeMdInline = (s) => DOMPurify.sanitize(marked.parseInline(s || ""));

function showView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  $("view-" + name).classList.add("active");
}

/* ---------------- upload ---------------- */
const dropzone = $("dropzone");
const fileInput = $("file-input");

$("browse-btn").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileInput.value = "";  // reset so re-selecting the same file after a failure still fires
  if (file) uploadFile(file);
});
["dragover", "dragenter"].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", e => {
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
// A drop anywhere outside the dropzone would otherwise navigate the tab to the file,
// destroying the session. Make stray drops inert.
window.addEventListener("dragover", e => e.preventDefault());
window.addEventListener("drop", e => e.preventDefault());

async function uploadFile(file) {
  $("upload-error").textContent = "";
  statusMisses = 0;
  showView("processing");
  $("processing-stage").textContent = "Uploading…";
  $("processing-file").textContent = file.name;

  const form = new FormData();
  form.append("file", file);
  let resp;
  try {
    resp = await fetch("/api/upload", { method: "POST", body: form });
  } catch (err) {
    return uploadFailed("Could not reach the server.");
  }
  if (!resp.ok) {
    const detail = (await resp.json().catch(() => ({}))).detail || "Upload failed.";
    return uploadFailed(detail);
  }
  const data = await resp.json();
  datasetId = data.dataset_id;
  pollStatus();
}

function uploadFailed(message) {
  showView("upload");
  $("upload-error").textContent = message;
}

async function pollStatus() {
  try {
    const resp = await fetch(`/api/status/${datasetId}`);
    if (resp.ok) {
      statusMisses = 0;
      const st = await resp.json();
      $("processing-stage").textContent = st.stage;
      if (st.done) {
        if (st.error) return uploadFailed("Analysis failed: " + st.error);
        return loadDashboard().catch(err =>
          uploadFailed("Could not render the report: " + err.message));
      }
    } else if (resp.status === 404 && ++statusMisses >= 3) {
      // Server lost this analysis (e.g. restarted). Keep retrying network/5xx (a
      // container may be rebooting), but a few 404s in a row is terminal.
      return uploadFailed("The server lost this analysis — please upload the file again.");
    }
  } catch (e) { /* transient — keep polling */ }
  setTimeout(pollStatus, 1200);
}

/* ---------------- dashboard ---------------- */
async function loadDashboard() {
  const resp = await fetch(`/api/report/${datasetId}`);
  if (!resp.ok) return uploadFailed("Could not load the report.");
  const r = await resp.json();

  $("ds-title").textContent = r.meta.filename;
  const totalRows = r.meta.tables.reduce((a, t) => a + t.rows, 0);
  $("ds-pill").textContent =
    `${r.meta.tables.length} table${r.meta.tables.length > 1 ? "s" : ""} · ${totalRows.toLocaleString()} rows · ` +
    (r.report_source === "llm" ? "AI summary" : "auto summary");

  $("summary").innerHTML = safeMd(r.summary_markdown);

  const qn = $("quality-notes");
  qn.innerHTML = "";
  (r.data_quality_notes || []).forEach(n => {
    const d = document.createElement("div");
    d.className = "qnote";
    d.innerHTML = safeMdInline(n);
    qn.appendChild(d);
  });

  const ins = $("insights");
  ins.innerHTML = "";
  (r.insights || []).forEach(i => {
    const d = document.createElement("div");
    d.className = `insight ${i.importance || "medium"}`;
    d.innerHTML = `<span class="tag">${i.importance || ""}</span><b></b><p></p>`;
    d.querySelector("b").textContent = i.title;
    d.querySelector("p").textContent = i.detail;
    ins.appendChild(d);
  });

  const chartsEl = $("charts");
  chartsEl.innerHTML = "";
  (r.charts || []).forEach(c => {
    const card = document.createElement("div");
    card.className = "chart-card";
    const box = document.createElement("div");
    box.className = "chart-box";
    const reason = document.createElement("div");
    reason.className = "chart-reason";
    reason.textContent = c.reason || "";
    card.appendChild(box); card.appendChild(reason);
    chartsEl.appendChild(card);
    renderChart(box, c.option);
  });

  const schemaEl = $("schema");
  schemaEl.innerHTML = "";
  (r.profiles || []).forEach(p => {
    const tn = document.createElement("div");
    tn.className = "tname";
    tn.textContent = `${p.table} — ${p.rows.toLocaleString()} rows`;
    schemaEl.appendChild(tn);
    const table = document.createElement("table");
    table.innerHTML = "<tr><th>Column</th><th>Type</th><th>Nulls</th><th>Unique</th><th>Notes</th></tr>";
    Object.entries(p.columns).forEach(([name, col]) => {
      const tr = document.createElement("tr");
      let notes = "";
      if (col.semantic_type === "numeric" && col.mean != null)
        notes = `min ${col.min} · mean ${col.mean} · max ${col.max}`;
      else if (col.top_values && col.top_values.length)
        notes = "top: " + col.top_values.slice(0, 3).map(v => v.value).join(", ");
      else if (col.min != null) notes = `${col.min} → ${col.max}`;
      tr.innerHTML = `<td>${name}</td><td><span class="badge ${col.semantic_type}">${col.semantic_type}</span></td>` +
        `<td>${col.null_pct}%</td><td>${col.unique.toLocaleString()}</td><td></td>`;
      tr.lastElementChild.textContent = notes;
      table.appendChild(tr);
    });
    schemaEl.appendChild(table);
  });

  showView("dashboard");
  // Charts were created while the dashboard was display:none (0x0); resize now that
  // it's visible so they lay out at full size.
  echartsInstances.forEach(c => c.resize());
}

function renderChart(el, option) {
  const chart = echarts.init(el, "dark", { renderer: "canvas" });
  option.backgroundColor = "transparent";
  chart.setOption(option);
  echartsInstances.push(chart);
}
window.addEventListener("resize", () => echartsInstances.forEach(c => c.resize()));

/* ---------------- chat ---------------- */
const chatForm = $("chat-form");
const chatText = $("chat-text");
const chatMessages = $("chat-messages");

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatText.value.trim();
  if (!message || !datasetId) return;
  chatText.value = "";
  addMsg("user", message);
  chatHistory.push({ role: "user", content: message });

  const pending = addMsg("assistant thinking", "Analyzing your data…");
  $("chat-send").disabled = true;
  try {
    const resp = await fetch(`/api/chat/${datasetId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: chatHistory.slice(0, -1) }),
    });
    const data = resp.ok ? await resp.json()
      : { answer: "The server returned an error — please try again.", charts: [] };
    pending.remove();
    renderAssistant(data);
    chatHistory.push({ role: "assistant", content: data.answer });
  } catch (err) {
    pending.remove();
    addMsg("assistant", "Network error — please try again.");
  } finally {
    $("chat-send").disabled = false;
    chatText.focus();
  }
});

function addMsg(cls, text) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

function renderAssistant(data) {
  const chartsById = {};
  (data.charts || []).forEach(c => { chartsById[c.id] = c; });

  const div = document.createElement("div");
  div.className = "msg assistant";
  div.style.whiteSpace = "normal";

  // split answer on [[chart:ID]] markers, render text as markdown + charts inline
  const parts = (data.answer || "").split(/\[\[chart:([A-Za-z0-9_-]+)\]\]/g);
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 0) {
      if (parts[i].trim()) {
        const md = document.createElement("div");
        md.className = "md";
        md.innerHTML = safeMd(parts[i].trim());
        div.appendChild(md);
      }
    } else {
      const c = chartsById[parts[i]];
      if (c) {
        const box = document.createElement("div");
        box.className = "chat-chart";
        div.appendChild(box);
        requestAnimationFrame(() => renderChart(box, c.option));
      }
    }
  }
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
