const state = {
  scenarios: null,
  draftText: "",
  draftTokens: [],
};

async function* sseStream(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    throw new Error(`${url} -> ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const chunk = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      if (chunk.startsWith("data:")) {
        const payload = chunk.slice(5).trim();
        if (payload) yield JSON.parse(payload);
      }
    }
  }
}

function el(id) {
  return document.getElementById(id);
}

async function loadScenarios() {
  const resp = await fetch("/scenarios");
  state.scenarios = await resp.json();
  const select = el("scenario-select");
  for (const s of state.scenarios.edits) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = `${s.label} (${s.category})`;
    select.appendChild(opt);
  }
  select.addEventListener("change", applyScenario);
  applyScenario();
}

function applyScenario() {
  const id = el("scenario-select").value;
  const s = state.scenarios.edits.find((x) => x.id === id);
  el("context-old").value = s.context_old;
  el("context-new").value = s.context_new;
  el("question").value = s.question;
  el("draft-pane").value = "";
  el("btn-edit").disabled = true;
  state.draftText = "";
  state.draftTokens = [];
  clearRace();
}

function clearRace() {
  for (const arm of ["baseline", "redraft"]) {
    el(`text-${arm}`).textContent = "";
    el(`readout-${arm}`).textContent = "";
    el(`timer-${arm}`).textContent = "0.00s";
  }
  el("headline").textContent = "";
}

async function generateDraft() {
  const context = el("context-old").value;
  const question = el("question").value;
  el("btn-draft").disabled = true;
  try {
    const resp = await fetch("/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context, question }),
    });
    const data = await resp.json();
    state.draftText = data.text;
    state.draftTokens = data.tokens;
    el("draft-pane").value = data.text;
    el("btn-edit").disabled = false;
  } finally {
    el("btn-draft").disabled = false;
  }
}

async function runEdit() {
  clearRace();
  el("btn-edit").disabled = true;
  const context = el("context-new").value;
  const question = el("question").value;
  const wallByArm = { baseline: 0, redraft: 0 };
  const doneByArm = { baseline: false, redraft: false };

  try {
    for await (const event of sseStream("/edit", {
      context,
      question,
      old_output_tokens: state.draftTokens,
    })) {
      const arm = event.arm;
      if (event.content) {
        el(`text-${arm}`).textContent += event.content;
      }
      el(`timer-${arm}`).textContent = `${event.t.toFixed(2)}s`;
      if (event.stop) {
        wallByArm[arm] = event.t;
        doneByArm[arm] = true;
        if (arm === "redraft") {
          const held = (event.redraft_held_fraction ?? 0).toFixed(3);
          const div = event.redraft_divergences ?? 0;
          el("readout-redraft").textContent = `held=${held} divergences=${div}`;
        }
      }
    }
  } finally {
    el("btn-edit").disabled = false;
  }

  if (doneByArm.baseline && doneByArm.redraft && wallByArm.redraft > 0) {
    const speedup = wallByArm.baseline / wallByArm.redraft;
    el("headline").textContent = `${speedup.toFixed(2)}x ${
      speedup >= 1 ? "faster" : "slower"
    } than baseline`;
  }
}

async function runBatch() {
  el("btn-batch").disabled = true;
  el("batch-body").innerHTML = "";
  el("batch-summary").textContent = "";
  try {
    for await (const event of sseStream("/batch", {})) {
      if (event.seed) continue;
      if (event.final) {
        el("batch-summary").textContent =
          `${event.steps} steps, cumulative speedup ${event.cumulative_speedup.toFixed(2)}x, ` +
          `${event.compute_saved_s.toFixed(2)}s saved, mean held ${event.mean_held_fraction.toFixed(3)}`;
        continue;
      }
      const row = document.createElement("tr");
      const stepSpeedup = event.baseline_wall_s / event.redraft_wall_s;
      const cells = [
        event.step,
        event.baseline_wall_s.toFixed(2),
        event.redraft_wall_s.toFixed(2),
        `${stepSpeedup.toFixed(2)}x`,
        event.held_fraction.toFixed(3),
      ];
      for (const value of cells) {
        const td = document.createElement("td");
        td.textContent = value;
        row.appendChild(td);
      }
      el("batch-body").appendChild(row);
    }
  } finally {
    el("btn-batch").disabled = false;
  }
}

el("btn-draft").addEventListener("click", generateDraft);
el("btn-edit").addEventListener("click", runEdit);
el("btn-batch").addEventListener("click", runBatch);
loadScenarios();
