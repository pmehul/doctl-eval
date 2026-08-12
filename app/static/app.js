/* The UI. No framework: one page and a few render functions, which is easier to
   read than a build pipeline. All the state is one `result` object, and that object
   is exactly what the API returns and exactly what gets saved to data/runs/. So
   anything you can see here, you can find in the saved file. */

const LABELS = ["bug", "enhancement", "question", "documentation", "security", "other"];
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let CATALOG = null, CORPUS = null, HEALTH = null, RESULT = null;
let scoredFilter = "disagree", unscoredFilter = "all";

/* ---------- formatting ---------- */
const pct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");
const f3 = (x) => (x == null ? "—" : Number(x).toFixed(3));
const ms = (x) => (x == null ? "—" : Math.round(x).toLocaleString() + " ms");
const secs = (x) => (x == null ? "—" : Number(x).toFixed(1) + " s");
const int = (x) => (x == null ? "—" : Number(x).toLocaleString());

// Costs here go from about $0.000001 for one cheap call up to a dollar or so for a
// whole run. One fixed number of decimals would either show every call as $0.00 or
// print a run total with nine decimal places, so it scales with the size.
function usd(x, opts = {}) {
  if (x == null) return "—";
  const v = Number(x);
  if (v === 0) return "$0";
  if (opts.per || Math.abs(v) < 0.01) return "$" + v.toPrecision(3);
  if (Math.abs(v) < 1) return "$" + v.toFixed(4);
  return "$" + v.toFixed(2);
}

/* ---------- boot ---------- */
async function boot() {
  [HEALTH, CATALOG, CORPUS] = await Promise.all([
    // cache: "no-store" on every one of these. The app is served from behind a
    // CDN, and /api/health decides whether the "these numbers are fake" banner
    // appears. A cached health response kept that banner on screen for a
    // deployment that had already been switched to real inference, which looks
    // exactly like the app ignoring its own configuration. The server sends
    // no-store as well; this is the browser half of the same fix.
    fetch("/api/health", { cache: "no-store" }).then((r) => r.json()),
    fetch("/api/catalog", { cache: "no-store" }).then((r) => r.json()),
    fetch("/api/corpus", { cache: "no-store" }).then((r) => r.json()),
  ]);

  renderMode();
  if (HEALTH.config_problems?.length || !HEALTH.corpus_ok) {
    const b = $("cfgBanner");
    b.hidden = false;
    b.innerHTML = "<strong>Not runnable.</strong><span>" +
      esc([...(HEALTH.config_problems || []), HEALTH.corpus_error].filter(Boolean).join(" · ")) +
      "</span>";
    $("runBtn").disabled = true;
  }

  $("corpusLine").innerHTML =
    `<code>${esc(CORPUS.repo)}</code> · ${int(CORPUS.n_issues)} issues, frozen ` +
    `${esc((CORPUS.frozen_at || "").slice(0, 10))} · corpus <code>${esc(CORPUS.corpus_hash)}</code> · ` +
    `${int(CORPUS.n_scored_test)} scored (test) / ${int(CORPUS.n_scored_dev)} dev / ` +
    `${int(CORPUS.n_unscored)} unscored · prompt <code>${esc(HEALTH.prompt_version)}</code>`;

  const opts = CATALOG.models.map((m) =>
    `<option value="${esc(m.id)}">${esc(m.label)} — ${esc(m.params)} — ` +
    `$${m.usd_per_m_input}/$${m.usd_per_m_output} per M</option>`).join("");
  $("modelA").innerHTML = opts;
  $("modelB").innerHTML = opts;
  $("modelA").value = CATALOG.default_model_a;
  $("modelB").value = CATALOG.default_model_b;
  $("concurrency").value = HEALTH.defaults.concurrency;
  $("scoredSplit").value = HEALTH.defaults.scored_split;

  renderMeta();
  renderMethod();
  await loadRuns();

  $("modelA").onchange = $("modelB").onchange = renderMeta;
  $("runBtn").onclick = startRun;

  document.querySelectorAll(".tab").forEach((t) => {
    t.onclick = () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      $("tab-" + t.dataset.tab).classList.add("active");
    };
  });
  wireChips("scoredFilters", (f) => { scoredFilter = f; renderScoredList(); });
  wireChips("unscoredFilters", (f) => { unscoredFilter = f; renderUnscoredList(); });

  // Load the last saved run straight away if there is one, so opening the app shows
  // results instead of an empty shell.
  try {
    const last = await fetch("/api/result", { cache: "no-store" });
    if (last.ok) { RESULT = await last.json(); renderAll(); }
  } catch { /* no run in this process yet */ }
}

/* Report the mode from the live health response, never from hardcoded text.
   Both the chip and the banner wording are built from HEALTH.provider here, so a
   stale page cannot claim PROVIDER=mock while the server is running for real. The
   chip is always visible, because a missing warning banner is also what a page
   that failed to load its JavaScript looks like, and those two situations should
   not look identical. */
function renderMode() {
  const chip = $("modeChip");
  const provider = HEALTH.provider || "unknown";

  if (HEALTH.simulated) {
    chip.className = "chip-mode chip-sim";
    chip.textContent = `SIMULATED · ${provider}`;
    chip.title = "Offline simulator. Nothing here is measured.";
    const b = $("simBanner");
    b.hidden = false;
    b.innerHTML =
      `<strong>These numbers are fake.</strong><span>The server reports ` +
      `<code>PROVIDER=${esc(provider)}</code>, which is the offline simulator. Labels, ` +
      `timings and token counts are invented. The cost arithmetic is real but it is ` +
      `being applied to made-up token counts. <em>Nothing on this screen proves ` +
      `anything.</em> Set <code>PROVIDER=digitalocean</code> and ` +
      `<code>DO_INFERENCE_API_KEY</code> for real numbers.</span>`;
  } else {
    chip.className = "chip-mode chip-live";
    chip.textContent = `LIVE · ${provider}`;
    chip.title = "Real inference calls. These numbers are measured.";
    $("simBanner").hidden = true;
    $("simBanner").innerHTML = "";
  }
}

function wireChips(containerId, fn) {
  const c = $(containerId);
  if (!c) return;
  c.querySelectorAll(".chip").forEach((chip) => {
    chip.onclick = () => {
      c.querySelectorAll(".chip").forEach((x) => x.classList.remove("active"));
      chip.classList.add("active");
      fn(chip.dataset.f);
    };
  });
}

function specOf(id) { return CATALOG.models.find((m) => m.id === id); }

function renderMeta() {
  for (const [sel, box] of [["modelA", "metaA"], ["modelB", "metaB"]]) {
    const m = specOf($(sel).value);
    if (!m) continue;
    $(box).innerHTML = `<dl>
      <dt>vendor</dt><dd>${esc(m.vendor)}</dd>
      <dt>params</dt><dd>${esc(m.params)}</dd>
      <dt>arch</dt><dd>${esc(m.architecture)}${m.reasoning ? " · reasoning" : ""}</dd>
      <dt>input</dt><dd>$${m.usd_per_m_input} / M tokens</dd>
      <dt>output</dt><dd>$${m.usd_per_m_output} / M tokens</dd>
      <dt>context</dt><dd>${int(m.context_window)}</dd>
    </dl><div class="why">${esc(m.why_included)}</div>`;
  }
}

/* ---------- run ---------- */
async function startRun() {
  const btn = $("runBtn");
  btn.disabled = true;
  $("progressWrap").hidden = false;
  $("runNote").textContent = "";

  const body = {
    model_a: $("modelA").value,
    model_b: $("modelB").value,
    concurrency: Number($("concurrency").value),
    scored_split: $("scoredSplit").value,
  };

  const poll = setInterval(async () => {
    try {
      const p = await (await fetch("/api/progress", { cache: "no-store" })).json();
      if (p.state === "running") {
        const frac = p.total ? p.completed / p.total : 0;
        $("barFill").style.width = (frac * 100).toFixed(1) + "%";
        $("progressText").textContent =
          `${p.completed} / ${p.total} calls · ${p.errors} errors · ` +
          `${p.throughput_rps.toFixed(1)} rps · elapsed ${secs(p.elapsed_s)} · eta ${secs(p.eta_s)}`;
      }
    } catch { /* transient */ }
  }, 400);

  try {
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    RESULT = data;
    $("barFill").style.width = "100%";
    $("progressText").textContent =
      `done · ${RESULT.operational.total_requests} calls in ${secs(RESULT.operational.wall_clock_s)} · ` +
      `saved as ${esc(RESULT.persisted_to || "(not persisted)")}`;
    renderAll();
    await loadRuns();
    document.querySelector('.tab[data-tab="scored"]').click();
  } catch (e) {
    $("runNote").innerHTML = `<span style="color:var(--bad)">${esc(e.message)}</span>`;
  } finally {
    clearInterval(poll);
    btn.disabled = false;
  }
}

async function loadRuns() {
  const { runs } = await (await fetch("/api/runs", { cache: "no-store" })).json();
  if (!runs.length) { $("runsTable").innerHTML = `<p class="hint">No persisted runs yet.</p>`; return; }
  $("runsTable").innerHTML = `<table><thead><tr>
      <th>finished</th><th>mode</th><th>model A</th><th>model B</th>
      <th>conc</th><th>split</th><th>A macro-F1</th><th>B macro-F1</th><th>wall</th><th></th>
    </tr></thead><tbody>${runs.map((r) => `<tr>
      <td>${esc((r.finished_at || "").replace("T", " ").slice(0, 16))}</td>
      <td>${r.simulated ? '<span class="tag tag-err">sim</span>' : '<span class="tag tag-gold">live</span>'}</td>
      <td class="mono">${esc(r.model_a)}</td>
      <td class="mono">${esc(r.model_b)}</td>
      <td>${int(r.concurrency)}</td>
      <td>${esc(r.scored_split)}</td>
      <td>${f3(r.a_macro_f1)}</td>
      <td>${f3(r.b_macro_f1)}</td>
      <td>${secs(r.wall_clock_s)}</td>
      <td><a href="#" data-file="${esc(r.file)}" class="loadrun">load</a></td>
    </tr>`).join("")}</tbody></table>`;
  document.querySelectorAll(".loadrun").forEach((a) => {
    a.onclick = async (ev) => {
      ev.preventDefault();
      RESULT = await (await fetch("/api/runs/" + encodeURIComponent(a.dataset.file), { cache: "no-store" })).json();
      renderAll();
      document.querySelector('.tab[data-tab="scored"]').click();
    };
  });
}

/* ---------- render ---------- */
function renderAll() {
  ["scored", "unscored", "ops"].forEach((k) => {
    $(k + "Empty").hidden = true;
    $(k + "Body").hidden = false;
  });
  renderHeadline();
  renderScored();
  renderUnscored();
  renderOps();
}

function nameA() { return RESULT.models.a.label; }
function nameB() { return RESULT.models.b.label; }

function renderHeadline() {
  const sa = RESULT.scored.a, sb = RESULT.scored.b;
  const oa = RESULT.operational.a, ob = RESULT.operational.b;
  const cheaper = (oa.cost.per_correct_usd ?? Infinity) <= (ob.cost.per_correct_usd ?? Infinity) ? "A" : "B";
  const better = sa.macro_f1 >= sb.macro_f1 ? "A" : "B";
  $("headlineStrip").innerHTML = `
    <div class="hl"><span class="v slot-a">${f3(sa.macro_f1)}</span><span class="k">A macro-F1</span></div>
    <div class="hl"><span class="v slot-b">${f3(sb.macro_f1)}</span><span class="k">B macro-F1</span></div>
    <div class="hl"><span class="v">${pct(RESULT.unscored.agreement.agreement_rate)}</span><span class="k">agreement</span></div>
    <div class="hl"><span class="v">${secs(RESULT.operational.wall_clock_s)}</span><span class="k">wall clock</span></div>
    <div class="hl"><span class="v">${better}/${cheaper}</span><span class="k">better / cheaper·correct</span></div>`;
}

/* --- scored --- */
function renderScored() {
  const sa = RESULT.scored.a, sb = RESULT.scored.b;
  const oa = RESULT.operational.a, ob = RESULT.operational.b;

  $("scoredSplitNote").innerHTML =
    `Split <code>${esc(RESULT.scored.split)}</code> · ${int(sa.n_scored)} issues · ` +
    `corpus <code>${esc(RESULT.corpus.corpus_hash)}</code>` +
    (RESULT.scored.split === "dev"
      ? ` · <span style="color:var(--warn)">dev is where the prompt was tuned; these scores are optimistic and should not be quoted</span>`
      : "");

  const mk = (slot, s, o, name) => `
    <div class="metric ${slot}">
      <div class="k">${slot.toUpperCase()} · ${esc(name)}</div>
      <div class="v">${f3(s.macro_f1)}</div>
      <div class="n">macro-F1 · excl. templated ${f3(s.macro_f1_excl_templated)}</div>
    </div>
    <div class="metric ${slot}">
      <div class="k">${slot.toUpperCase()} accuracy</div>
      <div class="v">${pct(s.accuracy)}</div>
      <div class="n">${int(s.correct)}/${int(s.n_usable)} · ${int(s.n_failed)} failed calls</div>
    </div>
    <div class="metric ${slot}">
      <div class="k">${slot.toUpperCase()} cost / correct</div>
      <div class="v">${usd(o.cost.per_correct_usd, { per: true })}</div>
      <div class="n">${usd(o.cost.per_call_usd, { per: true })} per call</div>
    </div>`;
  $("qualityHeadline").innerHTML = mk("a", sa, oa, nameA()) + mk("b", sb, ob, nameB());

  $("pcNameA").textContent = nameA();
  $("pcNameB").textContent = nameB();
  $("perClassA").innerHTML = perClassTable(sa.per_class);
  $("perClassB").innerHTML = perClassTable(sb.per_class);

  $("cmNameA").textContent = nameA();
  $("cmNameB").textContent = nameB();
  $("cmA").innerHTML = cmTable(sa.confusion_matrix);
  $("cmB").innerHTML = cmTable(sb.confusion_matrix);

  renderScoredList();
}

function perClassTable(pc) {
  return `<table><thead><tr><th>class</th><th>support</th><th>pred</th>
      <th>precision</th><th>recall</th><th>F1</th></tr></thead><tbody>${
    LABELS.map((l) => {
      const r = pc[l] || {};
      // Mark categories with too few examples to conclude anything from. With only
      // 3 of them, an F1 of 0.22 and an F1 of 0.80 are one issue apart. That's noise
      // and it should say so on screen, not be read off the table as a result.
      const thin = (r.support ?? 0) > 0 && (r.support ?? 0) < 10;
      return `<tr>
        <td>${esc(l)}${thin ? ' <span class="lbl" title="support under 10: this F1 is noise">thin</span>' : ""}</td>
        <td class="${thin ? "lowsupport" : ""}">${int(r.support)}</td>
        <td>${int(r.predicted)}</td>
        <td>${f3(r.precision)}</td><td>${f3(r.recall)}</td><td>${f3(r.f1)}</td></tr>`;
    }).join("")}</tbody></table>
    <p class="hint">“thin” means fewer than 10 examples in the answer key. One issue moves F1 by
    about 0.1 there, so a gap between two models means nothing.</p>`;
}

function cmTable(cm) {
  const rowTotals = {};
  LABELS.forEach((t) => { rowTotals[t] = LABELS.reduce((s, p) => s + (cm[t]?.[p] || 0), 0); });
  return `<table class="cm"><thead><tr><th>truth ↓ / pred →</th>${
    LABELS.map((l) => `<th class="rot">${esc(l.slice(0, 5))}</th>`).join("")}<th>n</th></tr></thead><tbody>${
    LABELS.map((t) => `<tr><td>${esc(t)}</td>${
      LABELS.map((p) => {
        const v = cm[t]?.[p] || 0;
        const cls = t === p ? "diag" : (v > 0 && v >= rowTotals[t] * 0.15 ? "hot" : (v === 0 ? "zero" : ""));
        return `<td class="${cls}">${v || "·"}</td>`;
      }).join("")}<td>${rowTotals[t]}</td></tr>`).join("")}</tbody></table>
    <p class="hint">Red cells are mistakes taking 15% or more of a category. Those are the
    boundaries worth fixing in the prompt, or routing around in production.</p>`;
}

function renderScoredList() {
  if (!RESULT) return;
  const items = RESULT.scored.items.filter((it) => {
    const aw = !it.a_correct, bw = !it.b_correct;
    switch (scoredFilter) {
      case "disagree": return it.models_disagree;
      case "either-wrong": return aw || bw;
      case "both-wrong": return aw && bw;
      case "a-wrong-b-right": return aw && !bw;
      case "b-wrong-a-right": return bw && !aw;
      default: return true;
    }
  });
  $("scoredCount").textContent =
    `${items.length} of ${RESULT.scored.items.length} scored issues · ground truth shown on every row`;
  $("scoredList").innerHTML = items.map((it) => `
    <div class="item">
      <div class="item-hd">
        <span class="item-num">#${it.number}</span>
        <span class="item-title">${esc(it.title)}</span>
        <a href="${esc(it.html_url)}" target="_blank" rel="noopener" class="small">github ↗</a>
      </div>
      <div class="item-labels">
        <span class="lbl">truth</span>
        <span class="tag tag-gold">${esc(it.gold_label)}</span>
        <span class="lbl" title="how this gold label was derived">${esc(it.gold_source)}</span>
        ${it.templated ? '<span class="lbl" title="bot-generated CVE report">templated</span>' : ""}
        <span class="lbl">A</span>${predTag(it.a, "a", it.a_correct)}
        <span class="lbl">B</span>${predTag(it.b, "b", it.b_correct)}
      </div>
      ${rawDetails(it)}
    </div>`).join("") || `<p class="hint">Nothing matches this filter.</p>`;
}

function predTag(side, slot, correct) {
  if (!side.label) {
    return `<span class="tag tag-err" title="${esc(side.error_detail || "")}">${esc(side.error_type || "failed")}</span>`;
  }
  const conf = side.confidence != null ? ` <span class="lbl">${side.confidence.toFixed(2)}</span>` : "";
  return `<span class="tag tag-${slot} ${correct == null ? "" : correct ? "tag-ok" : "tag-no"}">${esc(side.label)}</span>${conf}`;
}

function rawDetails(it) {
  const side = (s, name, slot) => `
    <p class="raw-h">${slot.toUpperCase()} · ${esc(name)} · ${ms(s.latency_ms)} ·
      ${int(s.prompt_tokens)}+${int(s.completion_tokens)} tok · ${usd(s.total_cost_usd, { per: true })}
      ${s.parse_strategy ? "· parsed via " + esc(s.parse_strategy) : ""}
      ${s.attempts > 1 ? "· " + s.attempts + " attempts" : ""}</p>
    <div class="raw">${esc(s.raw_output || s.error_detail || "(no output)")}</div>`;
  return `<details>
      <summary>raw model output &amp; issue body</summary>
      ${side(it.a, nameA(), "a")}
      ${side(it.b, nameB(), "b")}
      <p class="raw-h">issue body as sent to the model</p>
      <div class="raw">${esc(it.body || "(empty)")}</div>
    </details>`;
}

/* --- unscored --- */
function renderUnscored() {
  const ag = RESULT.unscored.agreement;
  $("agreementHeadline").innerHTML = `
    <div class="metric"><div class="k">agreement rate</div><div class="v">${pct(ag.agreement_rate)}</div>
      <div class="n">${int(ag.n_agreed)} of ${int(ag.n_both_succeeded)} both-succeeded</div></div>
    <div class="metric"><div class="k">disagreements</div><div class="v">${int(ag.n_disagreed)}</div>
      <div class="n">the escalation population</div></div>
    <div class="metric"><div class="k">unscored issues</div><div class="v">${int(ag.n)}</div>
      <div class="n">no ground truth available</div></div>`;

  const maxN = Math.max(1, ...LABELS.map((l) =>
    Math.max(ag.distribution_a[l] || 0, ag.distribution_b[l] || 0)));
  $("distChart").innerHTML = LABELS.map((l) => {
    const a = ag.distribution_a[l] || 0, b = ag.distribution_b[l] || 0;
    return `<div class="barrow"><span>${esc(l)}</span>
      <div><div class="track"><div class="seg-a" style="width:${(a / maxN) * 50}%"></div></div>
      <div class="track" style="margin-top:2px"><div class="seg-b" style="width:${(b / maxN) * 50}%"></div></div>
      <span class="num">A ${a} · B ${b}</span></div></div>`;
  }).join("") +
    `<p class="hint">Top bar <span class="slot-a">A</span>, bottom <span class="slot-b">B</span>.
     If one model's spread is much flatter or much spikier than the other's, it's drawing the
     lines in a different place. You'll see it here before anyone complains about it.</p>`;

  const pairs = Object.entries(ag.top_disagreement_pairs || {});
  $("disagreePairs").innerHTML = pairs.length
    ? `<table><thead><tr><th>A predicted vs B predicted</th><th>n</th><th>share of disagreements</th></tr></thead>
       <tbody>${pairs.map(([k, v]) =>
         `<tr><td class="mono">${esc(k)}</td><td>${v}</td><td>${pct(v / ag.n_disagreed)}</td></tr>`).join("")}
       </tbody></table>`
    : `<p class="hint">No disagreements.</p>`;

  renderUnscoredList();
}

function renderUnscoredList() {
  if (!RESULT) return;
  const items = RESULT.unscored.items.filter((it) => {
    if (unscoredFilter === "disagree") return it.models_disagree;
    if (unscoredFilter === "errors") return !it.a.label || !it.b.label;
    return true;
  });
  $("unscoredCount").textContent = `${items.length} of ${RESULT.unscored.items.length} unscored issues`;
  $("unscoredList").innerHTML = items.slice(0, 400).map((it) => `
    <div class="item">
      <div class="item-hd">
        <span class="item-num">#${it.number}</span>
        <span class="item-title">${esc(it.title)}</span>
        <a href="${esc(it.html_url)}" target="_blank" rel="noopener" class="small">github ↗</a>
      </div>
      <div class="item-labels">
        <span class="lbl">A</span>${predTag(it.a, "a", null)}
        <span class="lbl">B</span>${predTag(it.b, "b", null)}
        ${it.models_disagree ? '<span class="lbl" style="color:var(--warn)">disagree</span>' : ""}
        ${it.maintainer_labels.length ? `<span class="lbl">repo labels: ${esc(it.maintainer_labels.join(", "))}</span>` : ""}
      </div>
      ${rawDetails(it)}
    </div>`).join("") +
    (items.length > 400 ? `<p class="hint">Showing first 400 of ${items.length}. Full detail is in the persisted run JSON.</p>` : "");
}

/* --- ops --- */
function renderOps() {
  const o = RESULT.operational, oa = o.a, ob = o.b, c = RESULT.config;
  $("opsEnvelope").innerHTML = `
    <div class="metric"><div class="k">concurrency</div><div class="v">${int(c.concurrency)}</div>
      <div class="n">shared across both models</div></div>
    <div class="metric"><div class="k">wall clock</div><div class="v">${secs(o.wall_clock_s)}</div>
      <div class="n">${int(o.total_requests)} requests total</div></div>
    <div class="metric"><div class="k">throughput</div><div class="v">${o.aggregate_throughput_rps.toFixed(1)}</div>
      <div class="n">req/s sustained, aggregate</div></div>
    <div class="metric"><div class="k">total cost</div><div class="v">${usd(oa.cost.total_usd + ob.cost.total_usd)}</div>
      <div class="n">both models, this run</div></div>
    <div class="metric"><div class="k">temperature</div><div class="v">${c.temperature}</div>
      <div class="n">deterministic classification</div></div>` +
    (RESULT.simulated
      ? `<div class="metric"><div class="k">time scale</div><div class="v">${c.mock_time_scale}×</div>
         <div class="n">simulated; wall clock is compressed</div></div>` : "");

  const row = (k, av, bv, note = "") => `<tr><td>${k}${note ? `<span class="hint">${note}</span>` : ""}</td>
    <td>${av}</td><td>${bv}</td></tr>`;
  $("opsTable").innerHTML = `<table><thead><tr><th>metric</th>
      <th class="slot-a">A · ${esc(nameA())}</th><th class="slot-b">B · ${esc(nameB())}</th>
    </tr></thead><tbody>
    ${row("p50 latency", ms(oa.latency_ms.p50), ms(ob.latency_ms.p50), `measured at concurrency ${RESULT.config.concurrency}`)}
    ${row("p95 latency", ms(oa.latency_ms.p95), ms(ob.latency_ms.p95), `measured at concurrency ${RESULT.config.concurrency}`)}
    ${row("p99 latency", ms(oa.latency_ms.p99), ms(ob.latency_ms.p99))}
    ${row("max latency", ms(oa.latency_ms.max), ms(ob.latency_ms.max))}
    ${row("requests", int(oa.requests), int(ob.requests))}
    ${row("throughput (req/s)", oa.throughput_rps.toFixed(1), ob.throughput_rps.toFixed(1), "this model's share of the shared budget")}
    ${row("error rate", pct(oa.error_rate), pct(ob.error_rate))}
    ${row("retries", int(oa.retries), int(ob.retries))}
    ${row("mean prompt tokens", oa.tokens.mean_prompt.toFixed(0), ob.tokens.mean_prompt.toFixed(0))}
    ${row("mean completion tokens", oa.tokens.mean_completion.toFixed(1), ob.tokens.mean_completion.toFixed(1), "this is where reasoning models spend")}
    ${row("total tokens", int(oa.tokens.total), int(ob.tokens.total))}
    ${row("cost per call", usd(oa.cost.per_call_usd, { per: true }), usd(ob.cost.per_call_usd, { per: true }))}
    ${row("total cost", usd(oa.cost.total_usd), usd(ob.cost.total_usd))}
    ${row("cost per correct", usd(oa.cost.per_correct_usd, { per: true }), usd(ob.cost.per_correct_usd, { per: true }), "total cost ÷ how many it got right on the scored half")}
    ${row("projected cost / 1M issues", usd(oa.cost.per_call_usd * 1e6), usd(ob.cost.per_call_usd * 1e6), "this run's average cost per call, multiplied out")}
  </tbody></table>`;

  const kinds = [...new Set([...Object.keys(oa.errors_by_type), ...Object.keys(ob.errors_by_type)])];
  $("errTable").innerHTML = kinds.length
    ? `<table><thead><tr><th>type</th><th class="slot-a">A</th><th class="slot-b">B</th><th>remedy</th></tr></thead><tbody>${
      kinds.map((k) => `<tr><td class="mono">${esc(k)}</td>
        <td>${int(oa.errors_by_type[k] || 0)}</td><td>${int(ob.errors_by_type[k] || 0)}</td>
        <td style="text-align:left" class="hint">${esc(REMEDY[k] || "investigate")}</td></tr>`).join("")}
      </tbody></table>`
    : `<p class="hint">No failed calls in this run.</p>`;

  const sel = $("costIssue");
  sel.innerHTML = RESULT.scored.items.slice(0, 300).map((it) =>
    `<option value="${it.number}">#${it.number} — ${esc(it.title.slice(0, 70))}</option>`).join("");
  sel.onchange = renderCostTrace;
  renderCostTrace();
}

const REMEDY = {
  rate_limit: "drop the concurrency, back off longer, or ask for more quota",
  timeout: "raise REQUEST_TIMEOUT_S, or the model is too slow for this deadline",
  server_error: "retried on its own. If it keeps up, fail over",
  auth: "the key is wrong. Never retried, a bad key doesn't get better",
  bad_request: "malformed request, or a setting this model doesn't take",
  parse_error: "replied fine but with nothing usable in it. A prompt or model problem, not accuracy",
  network: "connection reset, DNS, TLS. Retried after a random wait",
  other: "not classified. See error_detail on the item",
};

function renderCostTrace() {
  const num = Number($("costIssue").value);
  const it = RESULT.scored.items.find((x) => x.number === num);
  if (!it) { $("costTrace").innerHTML = ""; return; }
  const block = (s, name, slot) => {
    if (s.total_cost_usd == null) {
      return `<div class="eq">${slot.toUpperCase()} · ${esc(name)}\nno cost: call failed (${esc(s.error_type)})</div>`;
    }
    const inC = s.prompt_tokens / 1e6 * s.usd_per_m_input;
    const outC = s.completion_tokens / 1e6 * s.usd_per_m_output;
    return `<div class="eq">${slot.toUpperCase()} · ${esc(name)}   (issue #${it.number})

  input :  ${int(s.prompt_tokens)} tok ÷ 1,000,000 × $${s.usd_per_m_input}/M  =  $${inC.toPrecision(6)}
  output:  ${int(s.completion_tokens)} tok ÷ 1,000,000 × $${s.usd_per_m_output}/M  =  $${outC.toPrecision(6)}
  ${"─".repeat(58)}
  total :  <span class="hi">$${Number(s.total_cost_usd).toPrecision(6)}</span>

  stored total_cost_usd = ${s.total_cost_usd}
  recomputed here       = ${inC + outC}
  latency ${ms(s.latency_ms)} · attempts ${s.attempts}${s.http_status ? " · HTTP " + s.http_status : ""}</div>`;
  };
  $("costTrace").innerHTML = block(it.a, nameA(), "a") + block(it.b, nameB(), "b") +
    `<p class="hint">Both lines use the same saved token counts and the same published rates. The
     token counts come from the provider, not from guessing locally, so this total matches the
     bill.</p>`;
}

/* --- methodology --- */
async function renderMethod() {
  const g = CORPUS.gold;
  $("methodCorpus").innerHTML = `
    <table><tbody>
      <tr><td>repository</td><td class="mono">${esc(CORPUS.repo)}</td></tr>
      <tr><td>issues in frozen snapshot</td><td>${int(CORPUS.n_issues)}</td></tr>
      <tr><td>corpus hash</td><td class="mono">${esc(CORPUS.corpus_hash)}</td></tr>
      <tr><td>frozen at</td><td>${esc(CORPUS.frozen_at)}</td></tr>
      <tr><td>gold total</td><td>${int(g.totals.gold)}</td></tr>
      <tr><td>&nbsp;&nbsp;from maintainer labels</td><td>${int(g.stats.tier_maintainer)}</td></tr>
      <tr><td>&nbsp;&nbsp;hand-adjudicated</td><td>${int(g.stats.tier_hand)}</td></tr>
      <tr><td>&nbsp;&nbsp;label conflicts resolved by precedence</td><td>${int(g.stats.conflicts_resolved)}</td></tr>
      <tr><td>excluded as genuinely ambiguous</td><td>${int(g.stats.excluded_ambiguous)}</td></tr>
      <tr><td>left unscored (no label available)</td><td>${int(g.stats.unlabeled_left_unscored)}</td></tr>
      <tr><td>dev / test</td><td>${int(g.totals.dev)} / ${int(g.totals.test)} (seed ${g.split.seed})</td></tr>
    </tbody></table>
    <p class="hint">Class distribution &mdash; test split: ${
      LABELS.map((l) => `${l} ${g.distribution.test[l] || 0}`).join(" · ")}</p>
    <p class="hint">All ${g.distribution.templated?.security || 0} <code>security</code> items are
    bot-generated CVE reports, flagged <code>templated</code>. See
    <code>data/ground_truth/ANNOTATION_GUIDE.md</code> for the decision rules and the honest
    account of what this gold set cannot support.</p>`;

  $("methodCatalog").innerHTML = `<table><thead><tr>
      <th>model</th><th>params</th><th>arch</th><th>$/M in</th><th>$/M out</th><th>why in the pool</th>
    </tr></thead><tbody>${CATALOG.models.map((m) => `<tr>
      <td class="mono">${esc(m.id)}</td><td>${esc(m.params)}</td>
      <td>${esc(m.architecture)}${m.reasoning ? " · reasoning" : ""}</td>
      <td>${m.usd_per_m_input}</td><td>${m.usd_per_m_output}</td>
      <td style="text-align:left" class="hint">${esc(m.why_included)}</td></tr>`).join("")}
    </tbody></table>
    <p class="hint">Rates from <a href="${esc(CATALOG.pricing_source)}" target="_blank" rel="noopener">DigitalOcean
    Inference pricing</a>, verified ${esc(CATALOG.pricing_verified)}.</p>`;

  const p = await (await fetch("/api/prompt", { cache: "no-store" })).json();
  $("methodPrompt").innerHTML = `
    <p class="raw-h">system prompt (<code>${esc(p.prompt_version)}</code>)</p>
    <div class="raw">${esc(p.system_prompt)}</div>
    <p class="raw-h">few-shot examples — all dev-split, enforced in code</p>
    <div class="raw">${esc(p.few_shot.map((m) => m.role + ": " + m.content).join("\n\n"))}</div>
    <p class="raw-h">full rendered request for issue #${p.example_issue_number}</p>
    <div class="raw">${esc(p.example_rendered_request.map((m) => m.role + ": " + m.content).join("\n\n"))}</div>`;
}

boot().catch((e) => {
  document.body.insertAdjacentHTML("afterbegin",
    `<div class="banner banner-error"><strong>UI failed to start.</strong><span>${esc(e.message)}</span></div>`);
});
