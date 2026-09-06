/* ============================================================
   CIVSA · Phase 1 client-side app
   Vanilla JS. Talks to the FastAPI backend on the same origin.
   ============================================================ */

const API = ""; // same-origin

/* ---------- Tab switching ---------- */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    document
      .querySelectorAll(".panel")
      .forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    document.getElementById(btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "vendors") loadVendors();
    if (btn.dataset.tab === "debug") loadDebugStats();
    if (btn.dataset.tab === "rankings") initRankings();
  });
});

/* ============================================================
   UPLOAD TAB
   ============================================================ */

// Populated on first upload from GET /api/labels — backend is source of truth.
let LABELS = [];

async function loadLabels() {
  if (LABELS.length) return; // already loaded
  try {
    const resp = await fetch("/api/labels");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    LABELS = Array.isArray(data.labels) ? data.labels : [];
    if (!LABELS.length) throw new Error("empty vocabulary");
  } catch (e) {
    console.error("[civsa] failed to load labels from /api/labels:", e);
    LABELS = ["other"]; // safe fallback so the UI still functions
  }
}

const fileInput = document.getElementById("file-input");
const stagingArea = document.getElementById("upload-staging");
const uploadLog = document.getElementById("upload-log");

if (fileInput) {
  fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    await loadLabels(); // ← ensure vocab is ready

    stagingArea.innerHTML = "";
    const card = makeFileCard(file);
    stagingArea.appendChild(card);
    stagingArea.hidden = false;
    stagingArea.scrollIntoView({ behavior: "smooth", block: "nearest" });
    fileInput.value = "";

    autoSuggestLabels(file, card);
  });
}

async function autoSuggestLabels(file, card) {
  const status = card.querySelector(".upload-status");
  status.classList.remove("error");
  status.textContent = "Auto-suggesting labels…";

  try {
    const form = new FormData();
    form.append("file", file);
    const resp = await fetch("/api/suggest-label", {
      method: "POST",
      body: form,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    // ← LOOK AT THIS IN THE BROWSER CONSOLE
    console.log("suggest-label response:", data);

    // Accept many possible shapes: {labels}, {label}, {suggestion}, {tags},
    // plain string, or plain array.
    let raw = [];
    if (Array.isArray(data)) raw = data;
    else if (typeof data === "string") raw = [data];
    else if (Array.isArray(data.labels)) raw = data.labels;
    else if (Array.isArray(data.tags)) raw = data.tags;
    else if (Array.isArray(data.suggested_labels))
      raw = data.suggested_labels; // future-proof
    else if (typeof data.label === "string") raw = [data.label];
    else if (typeof data.suggestion === "string") raw = [data.suggestion];
    else if (typeof data.suggested_label === "string")
      raw = [data.suggested_label]; // ← the fix
    else if (typeof data.tag === "string") raw = [data.tag];

    // Normalise: lowercase, trim, strip surrounding quotes/punctuation.
    const suggested = raw
      .map((s) =>
        String(s)
          .toLowerCase()
          .trim()
          .replace(/^["'`]+|["'`.,;]+$/g, ""),
      )
      .filter(Boolean);

    const boxes = card.querySelectorAll('.label-grid input[type="checkbox"]');
    const matched = [];
    const missed = [];
    boxes.forEach((cb) => {
      if (suggested.includes(cb.value.toLowerCase())) {
        cb.checked = true;
        cb.parentElement.classList.add("auto-suggested");
        matched.push(cb.value);
      }
    });
    suggested.forEach((s) => {
      if (!Array.from(boxes).some((cb) => cb.value.toLowerCase() === s))
        missed.push(s);
    });

    if (matched.length) {
      let msg = `Auto-suggested: ${matched.join(", ")}.`;
      if (missed.length) msg += ` (Not in vocabulary: ${missed.join(", ")}.)`;
      status.textContent = msg + " Adjust if needed before uploading.";
    } else if (suggested.length) {
      status.textContent = `Model suggested "${suggested.join(", ")}" — none match the 30-label vocabulary. Pick manually.`;
    } else {
      status.textContent =
        "No labels auto-suggested. Pick manually if you want.";
    }
  } catch (e) {
    status.textContent = `Auto-suggest failed: ${e.message}. You can still upload without labels.`;
  }
}

const browseBtn = document.getElementById("browse-btn");
if (browseBtn && fileInput) {
  browseBtn.addEventListener("click", () => fileInput.click());
}

function makeFileCard(file) {
  const card = document.createElement("div");
  card.className = "upload-card";

  const head = document.createElement("div");
  head.className = "upload-card-head";
  head.innerHTML = `
    <span class="doc-icon">📄</span>
    <div class="doc-meta">
      <div class="doc-name"></div>
      <div class="doc-size"></div>
    </div>
  `;
  head.querySelector(".doc-name").textContent = file.name;
  head.querySelector(".doc-size").textContent = formatBytes(file.size);
  card.appendChild(head);

  // Vendor row (NEW) — required by /api/upload
  const vendorRow = document.createElement("div");
  vendorRow.className = "vendor-row";
  vendorRow.innerHTML = `
    <label class="vendor-label" for="vendor-input">Vendor</label>
    <input type="text" class="vendor-input" id="vendor-input"
           placeholder="e.g. Nirmala Chemicals Pvt Ltd" autocomplete="off">
  `;
  const vendorInput = vendorRow.querySelector(".vendor-input");
  vendorInput.value = guessVendorFromFilename(file.name);
  card.appendChild(vendorRow);

  const lh = document.createElement("div");
  lh.className = "labels-heading";
  lh.textContent = "Assign one or more labels (optional):";
  card.appendChild(lh);

  const grid = document.createElement("div");
  grid.className = "label-grid";
  for (const label of LABELS) {
    const wrap = document.createElement("label");
    wrap.className = "label-checkbox";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = label;
    const span = document.createElement("span");
    span.textContent = label;
    wrap.appendChild(cb);
    wrap.appendChild(span);
    grid.appendChild(wrap);
  }
  card.appendChild(grid);

  const bar = document.createElement("div");
  bar.className = "progress-bar";
  bar.hidden = true;
  bar.innerHTML = '<div class="progress-fill"></div>';
  card.appendChild(bar);

  const status = document.createElement("div");
  status.className = "upload-status";
  card.appendChild(status);

  const actions = document.createElement("div");
  actions.className = "upload-actions";
  const uploadBtn = document.createElement("button");
  uploadBtn.className = "btn btn-primary";
  uploadBtn.textContent = "Upload";
  const cancelBtn = document.createElement("button");
  cancelBtn.className = "btn btn-ghost";
  cancelBtn.textContent = "Cancel";
  actions.appendChild(uploadBtn);
  actions.appendChild(cancelBtn);
  card.appendChild(actions);

  cancelBtn.addEventListener("click", () => closeStaging());
  uploadBtn.addEventListener("click", () =>
    uploadFile(file, { grid, bar, status, uploadBtn, cancelBtn, vendorInput }),
  );

  return card;
}

function guessVendorFromFilename(name) {
  let base = name.replace(/\.[^.]+$/, "");
  // strip trailing quote/invoice/rfq words and numbers
  base = base.replace(
    /[_\-\s]*(quote|quotation|quot|invoice|inv|contract|po|rfq|coa|msds|catalog|brochure)[_\-\s]*\d*$/i,
    "",
  );
  base = base
    .replace(/[_\-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return base.replace(/\b\w/g, (c) => c.toUpperCase());
}

function closeStaging() {
  stagingArea.innerHTML = "";
  stagingArea.hidden = true;
}

async function uploadFile(file, ui) {
  const vendor = (ui.vendorInput?.value || "").trim();
  if (!vendor) {
    ui.status.textContent = "Please enter a vendor name before uploading.";
    ui.status.classList.add("error");
    ui.vendorInput?.focus();
    return;
  }

  const labels = Array.from(
    ui.grid.querySelectorAll('input[type="checkbox"]:checked'),
  ).map((cb) => cb.value);

  ui.uploadBtn.disabled = true;
  ui.cancelBtn.disabled = true;
  ui.bar.hidden = false;
  ui.status.textContent = "Uploading…";
  ui.status.classList.remove("error");

  const fill = ui.bar.querySelector(".progress-fill");
  fill.style.width = "30%";

  try {
    const form = new FormData();
    form.append("file", file);
    form.append("vendor", vendor);
    // Backend requires at least one label; fall back to "other" so an empty
    // checkbox grid can never trigger a 422.
    const labelsToSend = labels.length ? labels : ["other"];
    for (const label of labelsToSend) form.append("labels", label);

    const resp = await fetch("/api/upload", { method: "POST", body: form });
    fill.style.width = "80%";

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(err || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    fill.style.width = "100%";

    closeStaging();

    if (data.status === "duplicate") {
      appendUploadLog({
        name: file.name,
        size: file.size,
        labels: labelsToSend,
        vendor: vendor,
        hash: data.hash,
        linkedTo: data.linked_to,
        duplicate: true,
      });
    } else {
      appendUploadLog({
        name: file.name,
        size: file.size,
        labels: data.labels || labelsToSend,
        vendor: vendor,
        path: data.path,
        hash: data.hash,
      });
    }
  } catch (e) {
    ui.status.textContent = `Upload failed: ${e.message}`;
    ui.status.classList.add("error");
    ui.uploadBtn.disabled = false;
    ui.cancelBtn.disabled = false;
    fill.style.width = "0%";
  }
}

function appendUploadLog(entry) {
  const empty = uploadLog.querySelector(".empty");
  if (empty) empty.remove();

  const row = document.createElement("div");
  row.className = entry.duplicate
    ? "upload-log-row duplicate"
    : "upload-log-row";

  row.innerHTML = `
    <span class="check">${entry.duplicate ? "↺" : "✓"}</span>
    <div class="log-body">
      <div class="log-line-primary"></div>
      <div class="log-line-secondary"></div>
      <div class="log-time" data-uploaded-at="${Date.now()}">just now</div>
    </div>
  `;

  const primary = row.querySelector(".log-line-primary");
  const nameStrong = document.createElement("strong");
  nameStrong.textContent = entry.name;
  primary.appendChild(nameStrong);
  primary.appendChild(document.createTextNode(entry.duplicate ? " " : " "));

  const verb = document.createElement("span");
  if (entry.duplicate) {
    verb.className = "log-verb-dup";
    verb.textContent = "already in corpus — nothing new indexed";
  } else {
    verb.textContent = "uploaded";
  }
  primary.appendChild(verb);

  if (entry.vendor) {
    const v = document.createElement("span");
    v.className = "log-vendor";
    v.textContent = ` → vendor: ${entry.vendor}`;
    primary.appendChild(v);
  }

  const parts = [];
  if (entry.duplicate) {
    if (entry.linkedTo) {
      // Show only the tail of the path so it isn't overwhelming
      const tail = String(entry.linkedTo).split("/").slice(-3).join("/");
      parts.push(`Existing copy: …/${tail}`);
    } else {
      parts.push("Existing copy in this vendor's folder");
    }
  } else {
    parts.push(
      entry.labels && entry.labels.length
        ? `Tagged: ${entry.labels.join(", ")}`
        : "No tags",
    );
  }
  parts.push(formatBytes(entry.size));
  if (entry.hash) parts.push(`sha256:${entry.hash.slice(0, 8)}…`);
  row.querySelector(".log-line-secondary").textContent = parts.join(" · ");

  uploadLog.insertBefore(row, uploadLog.firstChild);
}

// Tick relative timestamps in the log every 15s.
setInterval(() => {
  const now = Date.now();
  uploadLog.querySelectorAll(".log-time").forEach((el) => {
    const t = parseInt(el.dataset.uploadedAt || "0", 10);
    if (!t) return;
    const s = Math.floor((now - t) / 1000);
    if (s < 60) el.textContent = "just now";
    else if (s < 3600) el.textContent = `${Math.floor(s / 60)} min ago`;
    else if (s < 86400) el.textContent = `${Math.floor(s / 3600)} hr ago`;
    else el.textContent = `${Math.floor(s / 86400)} d ago`;
  });
}, 15000);

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

/* ============================================================
   ASK TAB
   ============================================================ */
const askform = document.getElementById("askform");
const queryIn = document.getElementById("query");
const answerBox = document.getElementById("answer");

document.querySelectorAll(".chip").forEach((c) => {
  c.addEventListener("click", () => {
    queryIn.value = c.dataset.q;
    askform.dispatchEvent(new Event("submit", { cancelable: true }));
  });
});

askform.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = queryIn.value.trim();
  if (!q) return;
  answerBox.innerHTML = `<div class="loading">Thinking...</div>`;

  const form = new FormData();
  form.append("q", q);
  form.append("user", "priya");

  try {
    const r = await fetch(`${API}/api/query`, { method: "POST", body: form });
    const body = await r.json();
    renderAnswer(q, body);
  } catch {
    answerBox.innerHTML = `<div class="card"><span class="badge refuse">Error</span>
                           <div class="body">Could not reach the backend. Is the API running?</div></div>`;
  }
});

const ROUTE_META = {
  structured_sql: { cls: "route-sql", label: "SQL fact" },
  structured_sql_empty: { cls: "route-sql-empty", label: "SQL · no rows" },
  structured_corpus: { cls: "route-sql", label: "Corpus lookup" },
  vendor_scoped: { cls: "route-vendor", label: "Vendor scoped" },
  rag: { cls: "route-rag", label: "Retrieved" },
  rag_empty: { cls: "route-rag-empty", label: "No relevant docs" },
  unrelated: { cls: "route-refuse", label: "Off-topic" },
};

function renderAnswer(question, body) {
  const card = document.createElement("div");
  card.className = "card";

  if (!body.allowed) {
    card.innerHTML = `
      <span class="route-badge route-refuse">Blocked · ${escapeHTML(body.reason || "rejected")}</span>
      <div class="body">${escapeHTML(body.message)}</div>`;
    answerBox.innerHTML = "";
    answerBox.appendChild(card);
    return;
  }

  const route = body.route_method || "unknown";
  const meta = ROUTE_META[route] || { cls: "route-unknown", label: route };
  const routeHtml = `<span class="route-badge ${meta.cls}">${escapeHTML(meta.label)}</span>`;

  const srcHtml =
    body.sources && body.sources.length
      ? `<div class="sources"><details><summary>Sources (${body.sources.length})</summary>
         <pre>${escapeHTML(JSON.stringify(body.sources, null, 2))}</pre></details></div>`
      : "";

  card.innerHTML = `
    ${routeHtml}
    <div class="body">${escapeHTML(body.answer).replace(/\n/g, "<br>")}</div>
    ${srcHtml}`;

  answerBox.innerHTML = "";
  answerBox.appendChild(card);
}

/* ============================================================
   VENDORS TAB (with edit + delete)
   ============================================================ */
async function loadVendors() {
  const list = document.getElementById("vendorlist");
  list.innerHTML = `<div class="loading">Loading...</div>`;
  try {
    const r = await fetch(`${API}/api/vendors`);
    const { vendors } = await r.json();
    if (vendors.length === 0) {
      list.innerHTML = `<div class="empty">No vendors yet. Upload a document to get started.</div>`;
      return;
    }
    list.innerHTML = "";
    for (const v of vendors) list.appendChild(await makeVendorCard(v));
  } catch {
    list.innerHTML = `<div class="empty">Could not load vendors. Is the API running?</div>`;
  }
}

async function makeVendorCard(vendor) {
  const details = document.createElement("details");
  details.className = "vendorcard";
  const summary = document.createElement("summary");
  summary.textContent = vendor;
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "vendor-body";
  body.innerHTML = `<div class="loading">Loading documents...</div>`;
  details.appendChild(body);

  details.addEventListener("toggle", async () => {
    if (!details.open) return;
    await renderVendorBody(vendor, body);
  });

  return details;
}

async function renderVendorBody(vendor, body) {
  body.innerHTML = `<div class="loading">Loading documents...</div>`;
  try {
    const r = await fetch(
      `${API}/api/vendor/${encodeURIComponent(vendor)}/manifest`,
    );
    const { folders } = await r.json();
    const docs = folders.flatMap((f) =>
      (f.docs || []).map((d) => ({ ...d, date: f.date })),
    );

    body.innerHTML = "";
    if (!docs.length) {
      body.innerHTML = `<div class="empty">No documents.</div>`;
    } else {
      for (const d of docs) {
        const row = document.createElement("div");
        row.className = "doc-row";
        row.dataset.filename = d.filename;
        row.dataset.date = d.date;
        row.innerHTML = `
          <div>
            <div><strong>${escapeHTML(d.filename)}</strong></div>
            <div class="fmeta">${escapeHTML(d.date)} · ${((d.size_bytes || 0) / 1024).toFixed(1)} KB</div>
          </div>
          <div style="text-align:right">
            <div class="doc-labels">${(d.labels || []).map(escapeHTML).join(" · ")}</div>
            <div style="display:flex;gap:4px;justify-content:flex-end;margin-top:6px">
              <button class="chip edit-labels-btn">Edit labels</button>
              <button class="chip danger delete-doc-btn">Delete</button>
            </div>
          </div>`;
        row
          .querySelector(".edit-labels-btn")
          .addEventListener("click", () => openLabelEditor(vendor, row, body));
        row
          .querySelector(".delete-doc-btn")
          .addEventListener("click", () =>
            deleteDoc(vendor, d.date, d.filename, body),
          );
        body.appendChild(row);
      }
    }

    const actions = document.createElement("div");
    actions.className = "vendor-actions";
    actions.innerHTML = `
      <button class="danger delete-vendor-btn">
        Delete this vendor (all ${docs.length} document${docs.length === 1 ? "" : "s"})
      </button>`;
    actions
      .querySelector(".delete-vendor-btn")
      .addEventListener("click", () => deleteVendor(vendor));
    body.appendChild(actions);
  } catch {
    body.innerHTML = `<div class="empty">Could not load documents.</div>`;
  }
}

function openLabelEditor(vendor, row, body) {
  const filename = row.dataset.filename;
  const date = row.dataset.date;
  const current = row
    .querySelector(".doc-labels")
    .textContent.split("·")
    .map((s) => s.trim())
    .filter(Boolean);

  const editor = document.createElement("div");
  editor.className = "doc-row";
  editor.innerHTML = `
    <div style="grid-column:1 / 3">
      <div style="font-size:12px;color:var(--ink-3);margin-bottom:8px">
        Editing labels for <strong>${escapeHTML(filename)}</strong>
        — click any number of labels
      </div>
      <div class="label-grid">
        ${LABELS.map(
          (l) => `
          <label class="label-checkbox">
            <input type="checkbox" value="${l}"${current.includes(l) ? " checked" : ""}>
            <span>${l}</span>
          </label>`,
        ).join("")}
      </div>
      <div style="display:flex;gap:8px;margin-top:12px;align-items:center">
        <button class="upload-btn save-btn">Save</button>
        <button class="chip cancel-btn">Cancel</button>
        <span class="status" style="margin-left:auto"></span>
      </div>
    </div>`;
  row.replaceWith(editor);

  const status = editor.querySelector(".status");

  editor.querySelector(".cancel-btn").onclick = () =>
    renderVendorBody(vendor, body);
  editor.querySelector(".save-btn").onclick = async () => {
    const newLabels = Array.from(
      editor.querySelectorAll('input[type="checkbox"]:checked'),
    ).map((cb) => cb.value);
    if (!newLabels.length) {
      status.textContent = "Pick at least one label.";
      status.style.color = "var(--accent)";
      return;
    }
    status.textContent = "Saving...";
    status.style.color = "var(--ink-3)";

    const form = new FormData();
    form.append("date", date);
    form.append("filename", filename);
    newLabels.forEach((l) => form.append("labels", l));

    try {
      const r = await fetch(
        `${API}/api/vendor/${encodeURIComponent(vendor)}/doc/labels`,
        { method: "PATCH", body: form },
      );
      if (r.ok) {
        status.textContent = "Saved.";
        status.style.color = "var(--verified)";
        setTimeout(() => renderVendorBody(vendor, body), 500);
      } else {
        status.textContent = "Error: " + (await r.text());
        status.style.color = "var(--accent)";
      }
    } catch {
      status.textContent = "Network error";
      status.style.color = "var(--accent)";
    }
  };
}

async function deleteDoc(vendor, date, filename, bodyEl) {
  const msg =
    `Delete "${filename}" from ${vendor} (${date})?\n\n` +
    "This removes the file, its manifest entry, and every indexed chunk.\n" +
    "This cannot be undone.";
  if (!confirm(msg)) return;

  const url =
    `${API}/api/vendor/${encodeURIComponent(vendor)}/doc` +
    `?date=${encodeURIComponent(date)}&filename=${encodeURIComponent(filename)}`;
  try {
    const r = await fetch(url, { method: "DELETE" });
    if (!r.ok) {
      alert("Delete failed: " + (await r.text()));
      return;
    }
    renderVendorBody(vendor, bodyEl);
  } catch {
    alert("Network error while deleting.");
  }
}

async function deleteVendor(vendor) {
  const msg =
    `Delete vendor "${vendor}" ENTIRELY?\n\n` +
    "All documents, all manifests, and every indexed chunk for this vendor will be permanently removed.\n\n" +
    "Type the vendor's name in the next box to confirm.";
  if (!confirm(msg)) return;
  const typed = prompt(`Type "${vendor}" to confirm deletion:`);
  if (typed !== vendor) {
    alert("Name did not match — cancelled.");
    return;
  }
  try {
    const r = await fetch(`${API}/api/vendor/${encodeURIComponent(vendor)}`, {
      method: "DELETE",
    });
    if (!r.ok) {
      alert("Delete failed: " + (await r.text()));
      return;
    }
    loadVendors();
  } catch {
    alert("Network error while deleting vendor.");
  }
}

setInterval(() => {
  if (!uploadLog) return;
  const now = Date.now();
  uploadLog.querySelectorAll(".log-time").forEach((el) => {
    const t = parseInt(el.dataset.uploadedAt || "0", 10);
    if (!t) return;
    const s = Math.floor((now - t) / 1000);
    if (s < 60) el.textContent = "just now";
    else if (s < 3600) el.textContent = `${Math.floor(s / 60)} min ago`;
    else if (s < 86400) el.textContent = `${Math.floor(s / 3600)} hr ago`;
    else el.textContent = `${Math.floor(s / 86400)} d ago`;
  });
}, 15000);

/* ============================================================
   DEBUG TAB
   ============================================================ */
async function loadDebugStats() {
  const box = document.getElementById("debug-stats");
  if (!box) return;
  box.innerHTML = `<div class="loading">Loading stats...</div>`;
  try {
    const r = await fetch(`${API}/api/debug/stats`);
    const s = await r.json();
    const vendorRows =
      Object.entries(s.chunks_per_vendor || {})
        .sort((a, b) => b[1] - a[1])
        .map(
          ([v, n]) =>
            `<tr><td>${escapeHTML(v)}</td><td style="text-align:right">${n}</td></tr>`,
        )
        .join("") ||
      `<tr><td colspan="2" class="empty">No vendors yet.</td></tr>`;
    box.innerHTML = `
      <div class="stat-grid">
        <div class="stat"><div class="stat-num">${s.chroma_chunks}</div><div class="stat-label">Chroma chunks</div></div>
        <div class="stat"><div class="stat-num">${s.tfidf_chunks}</div><div class="stat-label">TF-IDF chunks</div></div>
        <div class="stat"><div class="stat-num">${s.distinct_sources}</div><div class="stat-label">Documents</div></div>
      </div>
      <details style="margin-top:14px">
        <summary>Chunks per vendor</summary>
        <table class="debug-table"><thead><tr><th>Vendor</th><th style="text-align:right">Chunks</th></tr></thead><tbody>${vendorRows}</tbody></table>
      </details>
      <details>
        <summary>Sources (first 50)</summary>
        <pre>${(s.sources || []).map(escapeHTML).join("\n") || "(none)"}</pre>
      </details>`;
  } catch {
    box.innerHTML = `<div class="empty">Could not load stats. Is the API running?</div>`;
  }
}

const refreshBtn = document.getElementById("refresh-stats");
if (refreshBtn) refreshBtn.addEventListener("click", loadDebugStats);

const dqForm = document.getElementById("debug-query-form");
if (dqForm) {
  dqForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = document.getElementById("debug-query").value.trim();
    const labels = document.getElementById("debug-labels").value.trim();
    const box = document.getElementById("debug-query-results");
    if (!q) return;
    box.innerHTML = `<div class="loading">Probing both stores...</div>`;

    const form = new FormData();
    form.append("q", q);
    if (labels) form.append("label_filter", labels);

    try {
      const r = await fetch(`${API}/api/debug/query`, {
        method: "POST",
        body: form,
      });
      const body = await r.json();
      const render = (hits, scoreKey, scoreLabel, better) =>
        hits.length
          ? hits
              .map(
                (h, i) => `
              <div class="debug-hit">
                <div class="debug-hit-header">
                  <span class="rank">${i + 1}.</span>
                  <span class="score">${scoreLabel} = ${h[scoreKey].toFixed(3)}</span>
                  <span class="doc-labels">${escapeHTML(String(h.labels || ""))}</span>
                </div>
                <div class="fmeta">${escapeHTML(h.source)} · para ${h.para ?? "?"}</div>
                <div class="preview">${escapeHTML(h.text_preview)}</div>
              </div>`,
              )
              .join("")
          : `<div class="empty">No hits. (${better})</div>`;
      box.innerHTML = `
        <div class="debug-cols">
          <div><h4>TF-IDF · top 10 · higher score is better</h4>
               ${render(body.tfidf, "score", "score", "no keyword overlap")}</div>
          <div><h4>Chroma · top 10 · lower distance is better</h4>
               ${render(body.chroma, "distance", "dist", "corpus empty or all filtered out")}</div>
        </div>`;
    } catch {
      box.innerHTML = `<div class="empty">Probe failed.</div>`;
    }
  });
}

const dgForm = document.getElementById("debug-gate-form");
if (dgForm) {
  dgForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = document.getElementById("debug-gate-query").value.trim();
    const box = document.getElementById("debug-gate-results");
    if (!q) return;
    box.innerHTML = `<div class="loading">Probing gate...</div>`;
    const form = new FormData();
    form.append("q", q);
    try {
      const r = await fetch(`${API}/api/debug/gate`, {
        method: "POST",
        body: form,
      });
      const body = await r.json();
      const cls =
        body.final_verdict && body.final_verdict.startsWith("APPROVED")
          ? "intent"
          : "refuse";
      box.innerHTML = `
        <div class="card">
          <div class="badge ${cls}" style="margin-bottom:12px">
            ${escapeHTML(body.final_verdict || "unknown")}
          </div>
          <pre>${escapeHTML(JSON.stringify(body, null, 2))}</pre>
        </div>`;
    } catch {
      box.innerHTML = `<div class="empty">Probe failed.</div>`;
    }
  });
}

const vProbeBtn = document.getElementById("vendor-probe-btn");
const vProbeInp = document.getElementById("vendor-probe-input");
const vProbeOut = document.getElementById("vendor-probe-out");

if (vProbeBtn && vProbeInp && vProbeOut) {
  vProbeBtn.addEventListener("click", async () => {
    const q = (vProbeInp.value || "").trim();
    if (!q) return;
    vProbeOut.textContent = "Resolving…";
    try {
      const resp = await fetch(
        `/api/vendor-resolve?q=${encodeURIComponent(q)}`,
      );
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      vProbeOut.textContent = JSON.stringify(data, null, 2);
    } catch (e) {
      vProbeOut.textContent = `Error: ${e.message}`;
    }
  });
  vProbeInp.addEventListener("keydown", (e) => {
    if (e.key === "Enter") vProbeBtn.click();
  });
}

/* ---------- Chunk viewer ---------- */
const chunkBtn = document.getElementById("chunk-btn");
if (chunkBtn) {
  chunkBtn.addEventListener("click", async () => {
    const v = document.getElementById("chunk-vendor").value.trim();
    const d = document.getElementById("chunk-date").value.trim();
    const f = document.getElementById("chunk-filename").value.trim();
    const out = document.getElementById("chunk-out");
    if (!v || !d || !f) {
      out.textContent = "Fill vendor / date / filename.";
      return;
    }
    out.innerHTML = "Loading…";
    try {
      const resp = await fetch(
        `${API}/api/debug/chunks?vendor=${encodeURIComponent(v)}` +
          `&date=${encodeURIComponent(d)}&filename=${encodeURIComponent(f)}`,
      );
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      out.innerHTML = "";

      const summary = document.createElement("div");
      summary.className = "chunk-summary";
      summary.textContent = `${data.n_chunks} chunks total`;
      out.appendChild(summary);

      data.chunks.forEach((c, i) => {
        const el = document.createElement("div");
        const ctype = c.chunk_type || "unknown"; // flat dict — no c.metadata
        el.className = `chunk chunk-${ctype}`;
        const bits = [`#${i}`, ctype];
        if (c.row_index !== undefined) bits.push(`row ${c.row_index}`);
        if (c.prose_index !== undefined) bits.push(`prose ${c.prose_index}`);
        el.innerHTML =
          `<div class="chunk-head">${bits.join(" · ")}</div>` +
          `<pre>${escapeHTML(c.text)}</pre>`;
        out.appendChild(el);
      });
    } catch (e) {
      out.textContent = `Error: ${e.message}`;
    }
  });
}

/* ---------- Structured fact viewer ---------- */
const factsBtn = document.getElementById("facts-btn");
const factsInp = document.getElementById("facts-vendor");
const factsOut = document.getElementById("facts-out");

if (factsBtn && factsInp && factsOut) {
  factsBtn.addEventListener("click", async () => {
    const q = (factsInp.value || "").trim();
    if (!q) return;
    factsOut.innerHTML = `<div class="loading">Loading…</div>`;
    try {
      const r = await fetch(
        `${API}/api/debug/facts?vendor=${encodeURIComponent(q)}`,
      );
      if (!r.ok) throw new Error((await r.text()) || `HTTP ${r.status}`);
      const data = await r.json();
      factsOut.innerHTML = renderFacts(data);
    } catch (e) {
      factsOut.innerHTML = `<div class="empty">Error: ${escapeHTML(e.message)}</div>`;
    }
  });
  factsInp.addEventListener("keydown", (e) => {
    if (e.key === "Enter") factsBtn.click();
  });
}

function renderFacts(data) {
  const { canonical, resolve_method, facts } = data;
  const vf = facts.vendor_facts || [];
  const ct = facts.commercial_terms || [];
  const qi = facts.quote_items || [];

  const vfRow = vf[0] || {};
  const ctRow = ct[0] || {};

  const isoCell = (v) =>
    v === 1
      ? `<span class="iso-yes">✓ yes</span>`
      : v === 0
        ? `<span class="iso-no">✗ no</span>`
        : `<span class="iso-unk">— unknown</span>`;

  const kv = (label, val) =>
    `<tr><th>${escapeHTML(label)}</th><td>${val == null || val === "" ? "<em>—</em>" : escapeHTML(String(val))}</td></tr>`;

  const vendorFactsTbl = `
    <h4>vendor_facts (${vf.length} row${vf.length !== 1 ? "s" : ""})</h4>
    <table class="facts-table">
      ${kv("Canonical vendor", canonical)}
      <tr><th>ISO 9001</th><td>${isoCell(vfRow.iso_9001)}</td></tr>
      <tr><th>ISO 14001</th><td>${isoCell(vfRow.iso_14001)}</td></tr>
      <tr><th>ISO/IEC 17025</th><td>${isoCell(vfRow.iso_17025)}</td></tr>
      ${kv("Other certs", vfRow.other_certs)}
      ${kv("Address", vfRow.address)}
      ${kv("GSTIN", vfRow.gstin)}
      ${kv("PAN", vfRow.pan)}
      ${kv("CIN", vfRow.cin)}
      ${kv("Quote number", vfRow.quote_number)}
      ${kv("Quote date", vfRow.quote_date)}
      ${kv("Source", vfRow.source_doc)}
    </table>`;

  const termsTbl = `
    <h4>commercial_terms (${ct.length} row${ct.length !== 1 ? "s" : ""})</h4>
    <table class="facts-table">
      ${kv("Delivery (min days)", ctRow.delivery_days_min)}
      ${kv("Delivery (max days)", ctRow.delivery_days_max)}
      ${kv("Payment (days net)", ctRow.payment_days_net)}
      ${kv("Payment terms (raw)", ctRow.payment_terms_raw)}
      ${kv("Freight terms", ctRow.freight_terms)}
      ${kv("Warranty terms", ctRow.warranty_terms)}
      ${kv("GST %", ctRow.gst_percentage)}
      ${kv("Quote validity (days)", ctRow.quote_validity_days)}
    </table>`;

  const itemsTbl = qi.length
    ? `<h4>quote_items (${qi.length} row${qi.length !== 1 ? "s" : ""})</h4>
       <table class="facts-table items">
         <thead><tr><th>SKU</th><th>Description</th><th>Unit</th>
                    <th class="num">Qty</th><th class="num">Unit price (INR)</th>
                    <th class="num">Line total (INR)</th></tr></thead>
         <tbody>
         ${qi
           .map(
             (r) => `
           <tr>
             <td>${escapeHTML(r.sku || "")}</td>
             <td>${escapeHTML(r.description || "")}</td>
             <td>${escapeHTML(r.unit || "")}</td>
             <td class="num">${r.quantity ?? ""}</td>
             <td class="num">${r.unit_price_inr != null ? Math.round(r.unit_price_inr).toLocaleString("en-IN") : ""}</td>
             <td class="num">${r.line_total_inr != null ? Math.round(r.line_total_inr).toLocaleString("en-IN") : ""}</td>
           </tr>`,
           )
           .join("")}
         </tbody>
       </table>`
    : `<h4>quote_items (0 rows)</h4><div class="empty">No line items extracted.</div>`;

  return `
    <div class="facts-header">
      Resolved as <strong>${escapeHTML(canonical)}</strong>
      <span class="fmeta">(method: ${escapeHTML(resolve_method)})</span>
    </div>
    ${vendorFactsTbl}
    ${termsTbl}
    ${itemsTbl}`;
}

/* ============================================================
   RANKINGS TAB
   ============================================================ */
const rankBtn = document.getElementById("rank-btn");
const rankItemSingle = document.getElementById("rank-item-single");
const rankItemBasket = document.getElementById("rank-item-basket");
const rankResults = document.getElementById("rank-results");
const rankWeights = document.getElementById("rank-weights");

const DEFAULT_WEIGHTS = { S1: 0.35, S2: 0.2, S3: 0.2, S4: 0.15, S5: 0.1 };
const CRITERION_LABELS = {
  S1: "Price",
  S2: "Quality",
  S3: "Delivery",
  S4: "Financial Risk",
  S5: "ESG",
};
const CRITERION_ACTIVE = {
  S1: true,
  S2: false,
  S3: false,
  S4: false,
  S5: false,
};
let RANK_ITEMS = [];

async function initRankings() {
  if (RANK_ITEMS.length) return;
  try {
    const r = await fetch(`${API}/api/rankings/items`);
    const data = await r.json();
    RANK_ITEMS = data.items || [];
    populateItemPickers();
    populateWeights();
  } catch (e) {
    rankResults.innerHTML = `<div class="empty">Could not load items.</div>`;
  }
}

function populateItemPickers() {
  rankItemSingle.innerHTML = RANK_ITEMS.map(
    (it) => `<option value="${escapeHTML(it)}">${escapeHTML(it)}</option>`,
  ).join("");
  rankItemBasket.innerHTML = RANK_ITEMS.map(
    (it) => `<label class="basket-chip">
      <input type="checkbox" value="${escapeHTML(it)}">
      <span>${escapeHTML(it)}</span>
    </label>`,
  ).join("");
}

function populateWeights() {
  rankWeights.innerHTML = Object.keys(DEFAULT_WEIGHTS)
    .map(
      (k) => `
    <div class="weight-row ${CRITERION_ACTIVE[k] ? "" : "disabled"}">
      <label>${k} · ${CRITERION_LABELS[k]}
        ${CRITERION_ACTIVE[k] ? "" : '<span class="no-data">no data yet</span>'}
      </label>
      <input type="range" min="0" max="1" step="0.01"
             value="${DEFAULT_WEIGHTS[k]}" data-key="${k}"
             ${CRITERION_ACTIVE[k] ? "" : "disabled"}>
      <span class="weight-value" data-value="${k}">${DEFAULT_WEIGHTS[k].toFixed(2)}</span>
    </div>
  `,
    )
    .join("");
  rankWeights.querySelectorAll('input[type="range"]').forEach((inp) => {
    inp.addEventListener("input", (e) => {
      const k = e.target.dataset.key;
      rankWeights.querySelector(`[data-value="${k}"]`).textContent = parseFloat(
        e.target.value,
      ).toFixed(2);
    });
  });
}

function getCurrentWeights() {
  const out = {};
  rankWeights.querySelectorAll('input[type="range"]').forEach((inp) => {
    out[inp.dataset.key] = parseFloat(inp.value);
  });
  return out;
}

document.querySelectorAll('input[name="rank-mode"]').forEach((rb) => {
  rb.addEventListener("change", () => {
    const mode = rb.value;
    rankItemSingle.hidden = mode !== "single";
    rankItemBasket.hidden = mode !== "basket";
  });
});

if (rankBtn) {
  rankBtn.addEventListener("click", async () => {
    const mode = document.querySelector(
      'input[name="rank-mode"]:checked',
    ).value;
    let items = [];
    if (mode === "single") {
      items = [rankItemSingle.value];
    } else if (mode === "basket") {
      items = Array.from(
        rankItemBasket.querySelectorAll('input[type="checkbox"]:checked'),
      ).map((cb) => cb.value);
      if (!items.length) {
        rankResults.innerHTML = `<div class="empty">Pick at least one item.</div>`;
        return;
      }
    }
    rankResults.innerHTML = `<div class="loading">Computing…</div>`;
    try {
      const r = await fetch(`${API}/api/rankings/rank`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, items, weights: getCurrentWeights() }),
      });
      if (!r.ok) throw new Error((await r.text()) || `HTTP ${r.status}`);
      const data = await r.json();
      renderRanking(data);
    } catch (e) {
      rankResults.innerHTML = `<div class="empty">Ranking failed: ${escapeHTML(e.message)}</div>`;
    }
  });
}

function renderRanking(data) {
  if (!data.rows || !data.rows.length) {
    rankResults.innerHTML = `<div class="empty">No vendors quoted for the selected item(s).</div>`;
    return;
  }
  const critKeys = ["S1", "S2", "S3", "S4", "S5"];
  const critHeaders = critKeys
    .map((k) => `<th class="crit-col">${k}</th>`)
    .join("");

  const rows = data.rows
    .map((r) => {
      const critCells = critKeys
        .map((k) => {
          const v = r[k];
          if (v === null || v === undefined) {
            return `<td class="crit-col"><span class="crit-na">—</span></td>`;
          }
          const color = v >= 70 ? "good" : v >= 40 ? "ok" : "poor";
          return `<td class="crit-col">
        <div class="mini-bar mb-${color}" style="width:${Math.max(v, 2)}%"
             title="${k}: ${v}"></div>
      </td>`;
        })
        .join("");
      const priceCell = r.price
        ? `INR ${Number(r.price).toLocaleString("en-IN")}`
        : "—";
      return `
      <tr>
        <td>#${r.rank}</td>
        <td><strong>${escapeHTML(r.vendor)}</strong></td>
        <td class="score-col">${r.composite}</td>
        <td>${priceCell}</td>
        ${critCells}
        <td>
          <button class="btn btn-ghost btn-tiny detail-btn"
                  data-vendor="${escapeHTML(r.vendor)}">🔍</button>
        </td>
      </tr>`;
    })
    .join("");

  rankResults.innerHTML = `
    <table class="rank-table">
      <thead>
        <tr>
          <th>Rank</th><th>Vendor</th><th>Composite</th><th>Price</th>
          ${critHeaders}<th></th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <div id="rank-detail"></div>`;

  rankResults.querySelectorAll(".detail-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const vendor = btn.dataset.vendor;
      const row = data.rows.find((r) => r.vendor === vendor);
      renderRankDetail(row, data);
    });
  });
}

function renderRankDetail(row, data) {
  const detail = document.getElementById("rank-detail");
  const critKeys = ["S1", "S2", "S3", "S4", "S5"];
  const baseline = data.baseline || {};
  const contribs = row.contributions || {};

  const bars = critKeys
    .map((k) => {
      const c = contribs[k];
      if (c === null || c === undefined) {
        return `<div class="wf-row wf-na">
        <span class="wf-label">${k} · ${CRITERION_LABELS[k]}</span>
        <span class="wf-bar-na">no data</span>
      </div>`;
      }
      const cls = c > 0 ? "pos" : c < 0 ? "neg" : "zero";
      const arrow = c > 0 ? "↑" : c < 0 ? "↓" : "·";
      const word = c > 0 ? "better" : c < 0 ? "worse" : "neutral";
      const width = Math.min(Math.abs(c) * 3, 100);
      return `<div class="wf-row">
      <span class="wf-label">${k} · ${CRITERION_LABELS[k]}</span>
      <div class="wf-bar-container">
        <div class="wf-bar wf-${cls}" style="width:${width}%"></div>
      </div>
      <span class="wf-value ${cls}">
        ${arrow} ${word} · ${c > 0 ? "+" : ""}${c} pts
      </span>
    </div>`;
    })
    .join("");

  const baselineText =
    Object.entries(baseline)
      .filter(([k, v]) => v !== null)
      .map(([k, v]) => `${k}=${v}`)
      .join(", ") || "—";

  detail.innerHTML = `
    <div class="rank-detail-panel">
    <h3>${escapeHTML(row.vendor)} — score breakdown</h3>
    <div class="wf-legend">
      Composite ranges from 0 to 100 across the shortlist. Baseline is the
      average of the vendors who quoted this item — a positive bar means
      <strong>better than average</strong> for that criterion.
    </div>
    <div class="rank-detail-panel">
      <h3>${escapeHTML(row.vendor)} — score breakdown</h3>
      <div class="wf-baseline">
        Composite: <strong>${row.composite}</strong>
        &nbsp;·&nbsp; Baseline: ${baselineText}
      </div>
      <div class="wf-list">${bars}</div>
      <div class="rank-narrative">${escapeHTML(row.narrative || "")}</div>
      ${
        row.source
          ? `<div class="rank-source">Source: <code>${escapeHTML(row.source)}</code></div>`
          : ""
      }
    </div>`;
  detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ============================================================
   Utilities
   ============================================================ */
function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[c],
  );
}

const dropZone = document.getElementById("drop-zone");
if (dropZone && fileInput) {
  ["dragenter", "dragover"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragging");
    }),
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragging");
    }),
  );
  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer?.files;
    if (files && files.length) {
      // Route the first file through the same change-handler pipeline
      const dt = new DataTransfer();
      dt.items.add(files[0]);
      fileInput.files = dt.files;
      fileInput.dispatchEvent(new Event("change"));
    }
  });
}
