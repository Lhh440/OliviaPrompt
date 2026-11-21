/* OliviaPrompt web UI — vanilla JS, no build step.
 *
 * Flow: 生成槽位 (POST /api/ask) fills the editable slot table; every edit
 * re-renders via POST /api/render — the model never re-runs for formatting.
 */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  slots: {},        // name -> {value, src, impact, kind, options, tiers, doc}
  registry: [],     // from /api/slots
  activeTier: "long",
};

/* ---------------- api ---------------- */

async function api(path, body) {
  const opts = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) };
  const resp = await fetch(path, opts);
  const data = await resp.json().catch(() => ({ error: resp.statusText }));
  if (!resp.ok) throw new Error(data.error || resp.statusText);
  return data;
}

/* ---------------- slot table ---------------- */

function impactGroup(impact) {
  const wrap = document.createElement("div");
  wrap.className = "impact-group";
  const title = document.createElement("div");
  title.className = "impact-title";
  title.textContent = { high: "高影响 — 定错了整张图废掉", mid: "中影响", low: "低影响" }[impact];
  wrap.appendChild(title);
  return wrap;
}

function buildInput(def) {
  let el;
  if (def.kind === "bool") {
    el = document.createElement("select");
    for (const v of ["true", "false"]) {
      const o = document.createElement("option");
      o.value = v; o.textContent = v === "true" ? "室内 indoor" : "室外 outdoor";
      el.appendChild(o);
    }
    el.value = def.value ? "true" : "false";
  } else if (def.kind === "enum" && def.options.length) {
    el = document.createElement("select");
    for (const v of def.options) {
      const o = document.createElement("option");
      o.value = v; o.textContent = v;
      el.appendChild(o);
    }
    el.value = String(def.value);
  } else {
    el = document.createElement("input");
    el.type = "text";
    el.value = Array.isArray(def.value) ? def.value.join(", ") : String(def.value ?? "");
  }
  el.className = "slot-input";
  return el;
}

function buildSlotTable() {
  const groups = { high: null, mid: null, low: null };
  const root = $("slotGroups");
  root.innerHTML = "";
  const tier = $("tier").value;

  for (const def of state.registry) {
    const cur = state.slots[def.name] || { value: def.default, src: def.src || "default" };
    const row = document.createElement("div");
    row.className = "slot-row";
    row.dataset.name = def.name;
    row.title = def.doc || def.name;

    const name = document.createElement("span");
    name.className = "slot-name";
    name.textContent = def.name;

    const input = buildInput({ ...def, value: cur.value });
    input.addEventListener("input", () => onSlotEdit(def.name, input));
    input.addEventListener("change", () => onSlotEdit(def.name, input));

    const badge = document.createElement("span");
    badge.className = "badge";
    badge.dataset.src = cur.src;
    badge.textContent = { user: "用户", inferred: "推断", default: "默认" }[cur.src] || cur.src;

    row.append(name, input, badge);
    if (!groups[def.impact]) {
      groups[def.impact] = impactGroup(def.impact);
      root.appendChild(groups[def.impact]);
    }
    groups[def.impact].appendChild(row);
    updateRowState(row, def, tier);
  }
}

function updateRowState(row, def, tier) {
  row.classList.toggle("dimmed", !def.tiers.includes(tier));
  const src = state.slots[def.name]?.src || "default";
  row.classList.toggle(
    "needs-attention", def.impact === "high" && src === "default" && def.tiers.includes(tier)
  );
  const badge = row.querySelector(".badge");
  if (badge) {
    badge.dataset.src = src;
    badge.textContent = { user: "用户", inferred: "推断", default: "默认" }[src] || src;
  }
}

function collectSlots() {
  const payload = {};
  for (const def of state.registry) {
    const row = document.querySelector(`.slot-row[data-name="${def.name}"]`);
    if (!row) continue;
    const input = row.querySelector(".slot-input");
    let value;
    if (def.kind === "bool") value = input.value === "true";
    else if (def.kind === "enum") value = input.value;
    else if (def.kind === "list")
      value = input.value.split(",").map((s) => s.trim()).filter(Boolean);
    else if (def.name === "subject_count") value = parseInt(input.value, 10) || 1;
    else value = input.value;
    const src = state.slots[def.name]?.src || "default";
    payload[def.name] = { value, src };
  }
  return payload;
}

function onSlotEdit(name, input) {
  const cur = state.slots[name] || {};
  let value;
  const def = state.registry.find((d) => d.name === name);
  if (def?.kind === "bool") value = input.value === "true";
  else if (def?.kind === "enum") value = input.value;
  else if (def?.kind === "list")
    value = input.value.split(",").map((s) => s.trim()).filter(Boolean);
  else if (name === "subject_count") value = parseInt(input.value, 10) || 1;
  else value = input.value;
  state.slots[name] = { ...cur, value, src: "user" };
  scheduleRender();
}

/* ---------------- rendering ---------------- */

let renderTimer = null;
function scheduleRender() {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(() => render().catch(showError), 300);
}

async function render() {
  const data = await api("/api/render", {
    slots: collectSlots(),
    eco: $("eco").value,
    skeleton: $("skeleton").value,
  });
  state.slots = data.slots;
  for (const def of state.registry) {
    const row = document.querySelector(`.slot-row[data-name="${def.name}"]`);
    if (row) updateRowState(row, def, $("tier").value);
  }
  showOutputs(data.prompts, data.issues);
}

function showOutputs(prompts, issues) {
  const tier = state.activeTier;
  const block = prompts[tier];
  $("posText").textContent = block.positive;

  $("negText").textContent = block.negative || "（本生态留空）";
  const chip = $("negStrength");
  chip.textContent = { strong: "强有效", weak: "收益有限", none: "无效 → 转正向", param: "--no 参数" }[block.negative_strength];
  chip.className = "chip chip-" + block.negative_strength;
  $("negNote").textContent = block.negative_note || "";

  const hasPatch = (block.positive_patch || []).length > 0;
  $("patchBlock").hidden = !hasPatch;
  if (hasPatch) $("patchText").textContent = block.positive_patch.join(", ");

  const list = $("issues");
  list.innerHTML = "";
  if (!issues.length) {
    const li = document.createElement("li");
    li.className = "ok";
    li.textContent = "OK — 没有发现问题";
    list.appendChild(li);
  } else {
    for (const i of issues) {
      const li = document.createElement("li");
      li.className = i.severity;
      li.textContent = `${i.severity === "error" ? "ERROR" : "WARN"} ${i.slot}: ${i.message}`;
      list.appendChild(li);
    }
  }
}

function showError(err) {
  $("askNotes").innerHTML = "";
  const div = document.createElement("div");
  div.style.borderLeftColor = "var(--bad)";
  div.textContent = String(err.message || err);
  $("askNotes").appendChild(div);
}

/* ---------------- actions ---------------- */

async function ask() {
  const btn = $("btnAsk");
  btn.disabled = true;
  $("askNotes").innerHTML = "";
  try {
    const data = await api("/api/ask", {
      intent: $("intent").value,
      tier: $("tier").value,
      eco: $("eco").value,
      skeleton: $("skeleton").value,
      compact: $("compact").checked,
    });
    state.slots = data.slots;
    buildSlotTable();
    const notes = $("askNotes");
    notes.innerHTML = "";
    for (const n of data.notes || []) {
      const div = document.createElement("div");
      div.textContent = n;
      notes.appendChild(div);
    }
    await render();
  } catch (err) {
    showError(err);
  } finally {
    btn.disabled = false;
  }
}

/* ---------------- boot ---------------- */

async function boot() {
  const [registry, ecos, skeletons, cfg] = await Promise.all([
    api("/api/slots"), api("/api/eco"), api("/api/skeletons"), api("/api/config"),
  ]);
  state.registry = registry;

  const ecoSel = $("eco");
  for (const e of ecos) {
    const o = document.createElement("option");
    o.value = e.key;
    o.textContent = `${e.label}（负向${{ strong: "强", weak: "弱", none: "无效", param: "参数" }[e.strength]}）`;
    ecoSel.appendChild(o);
  }

  const skelSel = $("skeleton");
  for (const s of skeletons) {
    const o = document.createElement("option");
    o.value = s.key;
    o.textContent = `骨架 ${s.key} · ${s.doc}`;
    skelSel.appendChild(o);
  }

  const chip = $("modelChip");
  if (cfg.llm_ready) {
    chip.textContent = `模型已配置 · ${cfg.model}`;
    chip.className = "chip chip-strong";
    $("configHint").textContent = `端点 ${cfg.base_url} — 生成槽位只跑一次模型，之后改槽位即时重渲染。`;
  }

  $("btnAsk").addEventListener("click", ask);
  $("tier").addEventListener("change", () => buildSlotTable());
  $("tierTabs").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-tier]");
    if (!btn) return;
    state.activeTier = btn.dataset.tier;
    document.querySelectorAll("#tierTabs button[data-tier]").forEach((b) =>
      b.classList.toggle("active", b === btn));
    if (state.registry.length) render().catch(showError);
  });
  $("btnCopy").addEventListener("click", () =>
    navigator.clipboard.writeText($("posText").textContent));
  $("btnCopyNeg").addEventListener("click", () =>
    navigator.clipboard.writeText($("negText").textContent));

  // seed the table with defaults so the UI is useful without a model
  buildSlotTable();
}

boot().catch((err) => showError(err));
