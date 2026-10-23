// VectorMind — frontend. Talks only to the backend REST API.

const $ = (id) => document.getElementById(id);
const DIMS = 16;
const REGIONS = [["TECH", "tech"], ["SPACE", "space"], ["MUSIC", "music"], ["SPORTS", "sports"]];
const LEGEND = [["tech", "Tech"], ["space", "Space"], ["music", "Music"], ["sports", "Sports"], ["doc", "Documents (RAG)"]];
const KNOWN = new Set(["tech", "space", "music", "sports", "doc"]);

const state = {
  items: [],        // demo items {id, text, category, x, y, embedding}
  hits: new Set(),  // ids highlighted on the plot
  hoverId: null,    // id hovered on the plot or in the results list
  query: null,      // query point {x, y} on the plot
  results: [],
  algo: "hnsw",
  pulse: 0,
};

// ── helpers ──────────────────────────────────────────────────────────

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: options.body ? { "Content-Type": "application/json" } : {},
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(errorText(data) || `Request failed (${res.status})`);
  return data;
}

function errorText(data) {
  if (!data || !data.detail) return "";
  if (typeof data.detail === "string") return data.detail;
  return data.detail.map((d) => d.msg).join("; "); // Pydantic validation errors
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function color(cat) {
  return cssVar(`--c-${KNOWN.has(cat) ? cat : "custom"}`);
}

function chip(cat) {
  return el("span", { class: "chip", style: `--c:${color(cat)}` }, cat);
}

function fmtUs(us) {
  return us >= 1000 ? `${(us / 1000).toFixed(2)} ms` : `${us.toFixed(1)} µs`;
}

function showLeftError(msg) {
  $("left-error").textContent = msg || "";
  $("left-error").hidden = !msg;
}

// Weighted centre of the given item ids (rank 1 counts most), used to place the query star.
function centreOf(ids) {
  let sx = 0, sy = 0, sw = 0;
  ids.forEach((id, i) => {
    const p = state.items.find((it) => it.id === id);
    if (!p) return;
    const w = 1 / (i + 1);
    sx += p.x * w; sy += p.y * w; sw += w;
  });
  return sw ? { x: sx / sw, y: sy / sw } : null;
}

// ── tabs ─────────────────────────────────────────────────────────────

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("on", b === btn));
    document.querySelectorAll(".tab-body").forEach((p) =>
      p.classList.toggle("on", p.id === `tab-${btn.dataset.tab}`));
    if (btn.dataset.tab === "docs") loadDocs();
  });
});

// ── legend ───────────────────────────────────────────────────────────

function drawLegend() {
  $("legend").replaceChildren(...LEGEND.map(([cat, label]) =>
    el("li", {}, el("span", { class: "swatch", style: `--c:${color(cat)}` }), label)));
}

// ══════════════════════════════════════════════════════════════════════
//  OLLAMA STATUS
// ══════════════════════════════════════════════════════════════════════

async function loadStatus() {
  const badge = $("ollama-badge");
  const box = $("ollama-status");
  try {
    const s = await api("/status");
    const missing = [];
    if (!s.embed_model_installed) missing.push(s.embed_model);
    if (!s.gen_model_installed) missing.push(s.gen_model);
    const ready = s.ollama_available && !missing.length;
    badge.className = `badge ${ready ? "ok" : "err"}`;
    badge.textContent = ready ? "OLLAMA ✓" : "OLLAMA ✗";

    if (!s.ollama_available) {
      box.replaceChildren(
        el("div", { class: "off" }, "● Offline"),
        el("div", {}, "To enable documents and Ask AI:"),
        el("div", { class: "muted" }, "1. Install from ollama.com"),
        el("div", { class: "muted" }, `2. ollama pull ${s.embed_model}`),
        el("div", { class: "muted" }, `3. ollama pull ${s.gen_model}`),
      );
    } else {
      const lines = [
        el("div", { class: ready ? "on" : "off" }, ready ? "● Online" : `● Missing: ${missing.join(", ")}`),
        el("div", {}, "Embed: ", el("span", { style: "color:var(--accent)" }, s.embed_model)),
        el("div", {}, "Generate: ", el("span", { style: "color:var(--accent)" }, s.gen_model)),
        el("div", {}, "Dims: ", el("span", { class: "muted" }, s.doc_dims ?? "(set by the first insert)")),
        el("div", {}, "Chunks: ", s.doc_count),
      ];
      if (missing.length) lines.push(el("div", { class: "muted" }, `Run: ollama pull ${missing.join(" && ollama pull ")}`));
      box.replaceChildren(...lines);
    }
  } catch {
    badge.className = "badge err";
    badge.textContent = "BACKEND ✗";
    box.textContent = "Can't reach the backend.";
  }
}

// ══════════════════════════════════════════════════════════════════════
//  DEMO VECTORS
// ══════════════════════════════════════════════════════════════════════

async function loadItems() {
  const data = await api("/items");
  state.items = data.items;
  $("stats-label").textContent = `${data.items.length} vectors · ${data.dims} dims`;
}

async function loadHnsw() {
  const info = await api("/hnsw-info");
  const p = info.params;
  $("hnsw-params").textContent = `M=${p.M} · M0=${p.M0} · ef=${p.ef_search}`;
  const max = Math.max(info.nodes_per_layer[0] || 1, 1);
  const rows = info.nodes_per_layer.map((n, layer) =>
    el("div", { class: "bar-row" },
      el("div", { class: "lbl" },
        el("span", { class: "mono" }, `L${layer}`),
        el("span", { class: "v" }, `${n} nodes · ${info.edges_per_layer[layer]} edges`)),
      el("div", { class: "track" }, el("div", { class: "fill", style: `width:${(n / max) * 100}%` }))));
  $("layers").replaceChildren(...(rows.length ? rows : [el("p", { class: "muted small" }, "Empty")]));
}

async function refreshDemo() {
  await loadItems();
  await loadHnsw();
}

// algorithm toggle
document.querySelectorAll("#algo-seg button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("#algo-seg button").forEach((x) => x.classList.toggle("on", x === b));
    state.algo = b.dataset.algo;
  });
});

$("k").addEventListener("input", () => { $("k-label").textContent = $("k").value; });

$("search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("search-q").value.trim();
  if (!q) return;
  showLeftError("");
  const params = new URLSearchParams({ q, k: $("k").value, metric: $("metric").value, algo: state.algo });
  try {
    const data = await api(`/search?${params}`);
    state.results = data.results;
    state.hits = new Set(data.results.map((r) => r.id));
    state.query = { x: data.query.x, y: data.query.y };
    $("lat-big").textContent = fmtUs(data.latency_us);
    $("lat-sub").textContent = `${data.algo.toUpperCase()} · ${data.metric} · k=${data.k}`;
    renderResults();
    drawVecChart(data.query.vector);
    activateTab("search");
  } catch (err) {
    showLeftError(err.message);
  }
});

function activateTab(name) {
  document.querySelector(`.tab[data-tab="${name}"]`).click();
}

function renderResults() {
  const box = $("results");
  if (!state.results.length) {
    box.replaceChildren(el("p", { class: "muted small" }, "No results"));
    return;
  }
  box.replaceChildren(...state.results.map((r, i) =>
    el("div", {
      class: "rcard",
      onmouseenter: () => { state.hoverId = r.id; },
      onmouseleave: () => { state.hoverId = null; },
    },
      el("div", { class: "rank" }, `#${i + 1} NEAREST`),
      el("div", { class: "meta" }, r.text),
      el("div", { class: "foot" },
        chip(r.category),
        el("span", { class: "dist" }, `dist: ${r.distance.toFixed(5)}`),
        el("button", { class: "icon-btn", title: "Delete this item", onclick: () => deleteItem(r.id) }, "✕")))));
}

async function deleteItem(id) {
  try {
    await api(`/delete/${id}`, { method: "DELETE" });
    state.results = state.results.filter((r) => r.id !== id);
    state.hits.delete(id);
    renderResults();
    await refreshDemo();
  } catch (err) {
    showLeftError(err.message);
  }
}

$("insert-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showLeftError("");
  try {
    await api("/insert", {
      method: "POST",
      body: JSON.stringify({ text: $("insert-text").value.trim(), category: $("insert-cat").value }),
    });
    $("insert-text").value = "";
    await refreshDemo();
  } catch (err) {
    showLeftError(err.message);
  }
});

$("bench-btn").addEventListener("click", async () => {
  showLeftError("");
  const q = $("search-q").value.trim() || "python code algorithm";
  const params = new URLSearchParams({ q, k: $("k").value, metric: $("metric").value });
  try {
    const b = await api(`/benchmark?${params}`);
    const rows = [["Brute Force", "bruteforce", "--c-music"], ["KD-Tree", "kdtree", "--c-doc"], ["HNSW", "hnsw", "--c-space"]];
    const max = Math.max(...Object.values(b.timings_us), 1);
    $("bench-bars").replaceChildren(...rows.map(([label, key, c]) =>
      el("div", { class: "bar-row" },
        el("div", { class: "lbl" },
          el("span", {}, label),
          el("span", { class: "v" }, `${fmtUs(b.timings_us[key])} · recall ${Math.round(b.recall[key] * 100)}%`)),
        el("div", { class: "track" },
          el("div", { class: "fill", style: `width:${(b.timings_us[key] / max) * 100}%;--c:var(${c})` })))));
    $("bench-meta").textContent = `${b.item_count} vectors · k=${b.k} · ${b.metric} · median of ${b.repeats} runs`;
    $("bench-sec").hidden = false;
    activateTab("search");
  } catch (err) {
    showLeftError(err.message);
  }
});

// 16 bars, one per dimension, coloured by topic region.
function drawVecChart(vec) {
  const c = $("vec-chart");
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const pad = 6, base = h - 18, bw = (w - 2 * pad) / DIMS;
  for (let i = 0; i < DIMS; i++) {
    const bh = Math.max(1, vec[i] * (base - pad));
    g.fillStyle = color(REGIONS[Math.floor(i / 4)][1]);
    g.globalAlpha = 0.85;
    g.fillRect(pad + i * bw + 1, base - bh, bw - 2, bh);
  }
  g.globalAlpha = 1;
  g.font = "600 9.5px ui-monospace, Menlo, monospace";
  g.textAlign = "center";
  REGIONS.forEach(([label, cat], r) => {
    g.fillStyle = color(cat);
    g.fillText(label, pad + (r * 4 + 2) * bw, h - 5);
  });
}

// ══════════════════════════════════════════════════════════════════════
//  SCATTER PLOT (animated)
// ══════════════════════════════════════════════════════════════════════

const plot = $("plot");
const pctx = plot.getContext("2d");
let screenPts = [];

function frame() {
  const dpr = window.devicePixelRatio || 1;
  const w = plot.clientWidth, h = plot.clientHeight;
  if (plot.width !== Math.round(w * dpr) || plot.height !== Math.round(h * dpr)) {
    plot.width = Math.round(w * dpr);
    plot.height = Math.round(h * dpr);
  }
  const g = pctx;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  // grid + labels
  const P = 56;
  g.strokeStyle = cssVar("--grid");
  g.lineWidth = 1;
  for (let i = 0; i <= 8; i++) {
    const x = P + (i / 8) * (w - 2 * P), y = P + (i / 8) * (h - 2 * P);
    g.beginPath(); g.moveTo(x, P); g.lineTo(x, h - P); g.stroke();
    g.beginPath(); g.moveTo(P, y); g.lineTo(w - P, y); g.stroke();
  }
  const muted = cssVar("--muted");
  g.fillStyle = muted;
  g.font = "600 12px ui-monospace, Menlo, monospace";
  g.fillText("2D PCA projection · semantic space", 16, 26);
  g.font = "11px ui-monospace, Menlo, monospace";
  g.fillText("PC₁ →", w / 2 - 18, h - 18);
  g.save(); g.translate(20, h / 2 + 18); g.rotate(-Math.PI / 2); g.fillText("PC₂ →", 0, 0); g.restore();

  if (!state.items.length) {
    g.textAlign = "center";
    g.fillText("Connecting to VectorDB…", w / 2, h / 2);
    g.textAlign = "left";
    requestAnimationFrame(frame);
    return;
  }

  // bounds with 15% padding
  const pts = state.items.map((i) => [i.x, i.y]);
  if (state.query) pts.push([state.query.x, state.query.y]);
  let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
  for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  const px = (x1 - x0) * 0.15 || 0.1, py = (y1 - y0) * 0.15 || 0.1;
  x0 -= px; x1 += px; y0 -= py; y1 += py;
  const sx = (x) => P + ((x - x0) / (x1 - x0)) * (w - 2 * P);
  const sy = (y) => h - P - ((y - y0) / (y1 - y0)) * (h - 2 * P);

  // dashed lines from the query to its hits
  if (state.query && state.hits.size) {
    const qx = sx(state.query.x), qy = sy(state.query.y);
    g.strokeStyle = cssVar("--accent");
    g.globalAlpha = 0.4;
    g.setLineDash([4, 4]);
    for (const it of state.items) {
      if (!state.hits.has(it.id)) continue;
      g.beginPath(); g.moveTo(qx, qy); g.lineTo(sx(it.x), sy(it.y)); g.stroke();
    }
    g.setLineDash([]);
    g.globalAlpha = 1;
  }

  // points
  screenPts = [];
  for (const it of state.items) {
    const cx = sx(it.x), cy = sy(it.y);
    screenPts.push({ cx, cy, it });
    const col = color(it.category);
    const hit = state.hits.has(it.id);
    const r = hit ? 8 : 6;
    if (hit) { // pulsing ring
      g.beginPath();
      g.arc(cx, cy, r + 6 + Math.sin(state.pulse) * 3, 0, Math.PI * 2);
      g.strokeStyle = col; g.globalAlpha = 0.45; g.lineWidth = 1.5; g.stroke(); g.globalAlpha = 1;
    }
    g.beginPath();
    g.arc(cx, cy, r, 0, Math.PI * 2);
    g.fillStyle = col;
    g.globalAlpha = state.hits.size && !hit ? 0.4 : 1;
    g.fill();
    g.globalAlpha = 1;
    if (state.hoverId === it.id) {
      g.beginPath(); g.arc(cx, cy, r + 4, 0, Math.PI * 2);
      g.strokeStyle = col; g.lineWidth = 2; g.stroke();
    }
  }

  // query star
  if (state.query) {
    const qx = sx(state.query.x), qy = sy(state.query.y);
    g.save();
    g.translate(qx, qy);
    g.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = (i * Math.PI) / 5 - Math.PI / 2, rr = i % 2 === 0 ? 11 : 4.5;
      i ? g.lineTo(Math.cos(a) * rr, Math.sin(a) * rr) : g.moveTo(Math.cos(a) * rr, Math.sin(a) * rr);
    }
    g.closePath();
    g.fillStyle = cssVar("--text");
    g.fill();
    g.restore();
    g.fillStyle = muted;
    g.font = "600 11px ui-monospace, Menlo, monospace";
    const lw = g.measureText("query").width;
    g.fillText("query", Math.min(Math.max(qx - lw / 2, 4), w - lw - 4), qy + 25 > h - 4 ? qy - 16 : qy + 25);
  }

  state.pulse += 0.06;
  requestAnimationFrame(frame);
}

plot.addEventListener("mousemove", (e) => {
  const rect = plot.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = 14, near = null;
  for (const p of screenPts) {
    const d = Math.hypot(p.cx - mx, p.cy - my);
    if (d < best) { best = d; near = p; }
  }
  const tip = $("tooltip");
  state.hoverId = near ? near.it.id : null;
  if (!near) { tip.hidden = true; return; }
  tip.replaceChildren(el("div", { class: "cat" }, `[${near.it.category}] #${near.it.id}`), near.it.text);
  tip.style.left = `${Math.min(near.cx + 14, rect.width - 290)}px`;
  tip.style.top = `${near.cy + 12}px`;
  tip.hidden = false;
});
plot.addEventListener("mouseleave", () => { $("tooltip").hidden = true; state.hoverId = null; });

// ══════════════════════════════════════════════════════════════════════
//  DOCUMENTS
// ══════════════════════════════════════════════════════════════════════

async function loadDocs() {
  try {
    const data = await api("/doc/list");
    $("doc-count").textContent = data.count;
    if (!data.documents.length) {
      $("doc-list").replaceChildren(el("p", { class: "muted small" }, "No documents yet. Insert some above."));
      return;
    }
    $("doc-list").replaceChildren(...data.documents.map((d) =>
      el("div", { class: "dcard" },
        el("div", { class: "t" }, d.title),
        el("div", { class: "p" }, d.preview),
        el("div", { class: "f" },
          el("span", {}, `${d.words} words`),
          el("button", { class: "icon-btn", title: "Delete chunk", onclick: () => deleteDoc(d.id) }, "✕")))));
  } catch (err) {
    $("doc-list").replaceChildren(el("p", { class: "err-text small" }, err.message));
  }
}

async function deleteDoc(id) {
  try {
    await api(`/doc/delete/${id}`, { method: "DELETE" });
  } finally {
    loadDocs(); loadStatus(); refreshDemo();
  }
}

$("doc-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const title = $("doc-title").value.trim(), text = $("doc-text").value.trim();
  const btn = $("doc-btn"), status = $("doc-status");
  if (!title || !text) { status.className = "small err-text"; status.textContent = "Need both a title and text."; return; }
  btn.disabled = true;
  btn.textContent = "Embedding…";
  status.className = "small muted";
  status.textContent = "Calling Ollama nomic-embed-text…";
  try {
    const r = await api("/doc/insert", { method: "POST", body: JSON.stringify({ title, text }) });
    status.className = "small ok-text";
    status.textContent = `✓ Inserted ${r.chunks} chunk(s) · ${r.dims}D embeddings`;
    $("doc-title").value = "";
    $("doc-text").value = "";
    loadDocs(); loadStatus(); refreshDemo();
  } catch (err) {
    status.className = "small err-text";
    status.textContent = `✗ ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Embed & insert";
  }
});

// ══════════════════════════════════════════════════════════════════════
//  ASK AI (RAG, streamed)
// ══════════════════════════════════════════════════════════════════════

$("ask-q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("ask-form").requestSubmit();
});

// Highlight the retrieved documents on the map, or, if nothing was
// retrieved, still drop the query star into the demo space.
async function showRetrievalOnMap(contexts, question) {
  const ids = [...new Set(contexts.map((c) => c.marker_id).filter((id) => id != null))];
  if (ids.length) {
    state.hits = new Set(ids);
    state.query = centreOf(ids);
    return;
  }
  state.hits = new Set();
  try {
    const d = await api(`/search?${new URLSearchParams({ q: question, k: 3 })}`);
    state.query = { x: d.query.x, y: d.query.y };
  } catch {
    state.query = null; // no known keywords: nothing meaningful to plot
  }
}

function contextBlock(contexts) {
  const box = el("div", { class: "ctx" },
    el("div", { class: "label" }, `RETRIEVED CONTEXT (${contexts.length} chunk${contexts.length === 1 ? "" : "s"})`));
  if (!contexts.length) {
    box.append(el("div", { class: "muted small" }, "No document was close enough to the question (cosine distance ≤ 0.7), so the answer comes from the model's general knowledge."));
    return box;
  }
  const detail = el("div", { class: "ctx-text" });
  detail.hidden = true;
  const chips = contexts.map((c) => {
    const btn = el("button", { class: "ctx-chip", type: "button" }, `#${c.rank} ${c.title} · ${c.distance.toFixed(3)}`);
    btn.addEventListener("click", () => {
      const open = btn.classList.contains("sel");
      chips.forEach((b) => b.classList.remove("sel"));
      detail.hidden = open;
      if (!open) { btn.classList.add("sel"); detail.textContent = c.text; }
    });
    return btn;
  });
  box.append(...chips, detail);
  return box;
}

$("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("ask-q").value.trim();
  if (!question) return;
  const btn = $("ask-btn"), chat = $("chat");
  btn.disabled = true;
  btn.textContent = "Thinking…";

  const thinking = el("div", { class: "thinking" }, el("div", { class: "spinner" }), "Retrieving context & generating answer…");
  chat.replaceChildren(el("div", { class: "bubble-q" }, question), thinking);

  let textEl = null;
  try {
    const res = await fetch("/doc/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: Number($("ask-k").value) }),
    });
    if (!res.ok) throw new Error(errorText(await res.json().catch(() => ({}))) || `Request failed (${res.status})`);

    // newline-delimited JSON events: contexts, token…, done | error
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "contexts") {
          thinking.remove();
          textEl = el("div", { class: "text typing" });
          chat.append(el("div", { class: "bubble-a" },
            el("div", { class: "label" }, `🤖 ${ev.model}`), textEl, contextBlock(ev.contexts)));
          showRetrievalOnMap(ev.contexts, question);
        } else if (ev.type === "token") {
          textEl.textContent += ev.text;
        } else if (ev.type === "error") {
          throw new Error(ev.message);
        }
      }
    }
    $("ask-q").value = "";
  } catch (err) {
    thinking.remove();
    chat.append(el("div", { class: "bubble-a" },
      el("div", { class: "label" }, "ERROR"), el("div", { class: "text err-text" }, err.message)));
  } finally {
    if (textEl) textEl.classList.remove("typing");
    btn.disabled = false;
    btn.textContent = "Ask AI";
  }
});

// ── boot ─────────────────────────────────────────────────────────────

drawLegend();
drawVecChart(new Array(DIMS).fill(0));
requestAnimationFrame(frame);
refreshDemo().catch((err) => showLeftError(err.message));
loadStatus();
loadDocs();
setInterval(loadStatus, 15000);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  drawLegend();
  if (state.results.length) renderResults();
});
