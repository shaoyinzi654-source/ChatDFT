/* ==========================================================================
   ChatDFT front end.
   Talks to the FastAPI backend, renders SCF results, drives 3Dmol.js and
   Chart.js. No framework, no build step - just careful DOM work.
   ========================================================================== */

'use strict';

const API = '';

/* ------------------------------------------------------------------ state */
const state = {
  methods: null,
  molecule: null,
  lastResult: null,
  activeJob: null,
  pollTimer: null,
  viewer: null,
  style: 'ballstick',
  spin: false,
  charts: { orb: null, spec: null, scan: null, nci: null, series: null, uv: null,
    raman: null, nmr: null },
  // which ordinate the absorption figure uses: 'relative' (0-100, the shape)
  // or 'epsilon' (absolute L mol-1 cm-1, the number a table quotes)
  uvScale: 'relative',
  jobs: [],
  series: null,
  seriesProp: null,
  seriesProps: [],
  field: null,
  surfaces: null,
  cubeCache: null,
  surfWired: false,
  nci: null,
  nciWired: false,
};

const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ------------------------------------------------------------------ utils */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}

function fmt(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '&ndash;';
  if (typeof v !== 'number') return String(v);
  if (v !== 0 && Math.abs(v) < 1e-4) return v.toExponential(2);
  return v.toFixed(digits);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/** Minimal, safe markdown -> HTML for assistant replies. */
function markdown(src) {
  if (!src) return '';
  let out = escapeHtml(src);

  // fenced code
  out = out.replace(/```(\w*)\n([\s\S]*?)```/g,
    (m, lang, body) => `<pre><code>${body}</code></pre>`);

  // tables (| a | b |)
  out = out.replace(/(^\|.*\|\s*$\n?)+/gm, (block) => {
    const rows = block.trim().split('\n').filter(Boolean);
    if (rows.length < 2) return block;
    const head = rows[0].split('|').slice(1, -1).map((c) => c.trim());
    const bodyRows = rows.slice(2).map((r) => r.split('|').slice(1, -1).map((c) => c.trim()));
    let html = '<table><thead><tr>' +
      head.map((h) => `<th>${h}</th>`).join('') + '</tr></thead><tbody>';
    html += bodyRows.map((r) =>
      '<tr>' + r.map((c) => `<td>${c}</td>`).join('') + '</tr>').join('');
    return html + '</tbody></table>';
  });

  // headings
  out = out.replace(/^###\s+(.+)$/gm, '<h4>$1</h4>');
  out = out.replace(/^##\s+(.+)$/gm, '<h4>$1</h4>');

  // bold / italic / inline code
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');

  // lists
  out = out.replace(/(?:^[-*]\s+.+$\n?)+/gm, (block) => {
    const items = block.trim().split('\n')
      .map((l) => l.replace(/^[-*]\s+/, ''));
    return '<ul>' + items.map((i) => `<li>${i}</li>`).join('') + '</ul>';
  });
  out = out.replace(/(?:^\d+\.\s+.+$\n?)+/gm, (block) => {
    const items = block.trim().split('\n')
      .map((l) => l.replace(/^\d+\.\s+/, ''));
    return '<ol>' + items.map((i) => `<li>${i}</li>`).join('') + '</ol>';
  });

  // paragraphs
  out = out.split(/\n{2,}/).map((para) => {
    const t = para.trim();
    if (!t) return '';
    if (/^<(ul|ol|table|h4|pre)/.test(t)) return t;
    return `<p>${t.replace(/\n/g, '<br>')}</p>`;
  }).join('');

  return out;
}

/* ------------------------------------------------------------------ chat */
const chatScroll = $('#chatScroll');

function addMessage(role, html, extras) {
  const wrap = el('div', `msg ${role}`);
  const bubble = el('div', 'bubble');
  bubble.innerHTML = html;
  if (extras) bubble.appendChild(extras);
  wrap.appendChild(bubble);
  chatScroll.appendChild(wrap);
  chatScroll.scrollTop = chatScroll.scrollHeight;
  return bubble;
}

function addThinking(text) {
  const wrap = el('div', 'msg assistant');
  wrap.dataset.thinking = '1';
  const bubble = el('div', 'bubble');
  bubble.innerHTML = `<div class="thinking"><span class="spinner"></span>` +
    `<span>${escapeHtml(text)}</span></div>`;
  wrap.appendChild(bubble);
  chatScroll.appendChild(wrap);
  chatScroll.scrollTop = chatScroll.scrollHeight;
  return wrap;
}

function removeThinking() {
  $$('[data-thinking="1"]').forEach((n) => n.remove());
}

function renderIntentCard(intent, molecule, planner) {
  const card = el('div', 'intent-card');
  const rows = [
    ['Job', intent.job_type.replace(/_/g, ' ')],
    ['Molecule', molecule ? `${molecule.name || molecule.formula}` : intent.molecule],
    ['Method', `${intent.functional_label || intent.functional} / ${intent.basis_label || intent.basis}`],
    ['Charge &middot; spin', `${intent.charge} &middot; ${intent.multiplicity}`],
  ];
  if (intent.solvation) rows.push(['Solvent', `eps = ${intent.solvation}`]);
  // Which of the two planners read the sentence. Without this the reply looks
  // identical whether a model understood it or a regular expression matched.
  const notes = intent.notes || [];
  const named = notes.find((n) => /^planned by /.test(n));
  if (planner) {
    rows.push(['Planned by', planner === 'llm'
      ? (named ? named.replace(/^planned by /, '') : 'the language model')
      : 'local keyword parser']);
  }
  card.innerHTML = rows.map(([k, v]) =>
    `<div class="row"><span class="k">${k}</span><span class="v">${escapeHtml(v)}</span></div>`
  ).join('');
  return card;
}

/* ------------------------------------------------------------------ API */
async function api(path, opts) {
  const res = await fetch(API + path, Object.assign({
    headers: { 'Content-Type': 'application/json' },
  }, opts));
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
  if (!res.ok) {
    throw new Error(data.detail || data.message || `HTTP ${res.status}`);
  }
  return data;
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  try {
    const health = await api('/api/health');
    setEngineStatus(health);
  } catch (e) {
    setEngineStatus(null, e.message);
  }
  // Whether a language model is behind the planner is not decoration: with no
  // key the chat is a keyword parser, and the interface said nothing about it,
  // so "is there any AI in this?" was a fair question with no visible answer.
  try {
    setModelStatus(await api('/api/llm'));
  } catch (e) {
    setModelStatus(null, e.message);
  }
  try {
    state.methods = await api('/api/methods');
    populateSelects();
  } catch (e) {
    addMessage('assistant', `<div class="error-box">Could not load methods: ${escapeHtml(e.message)}</div>`);
  }
  try {
    const lib = await api('/api/library');
    renderLibrary(lib.molecules);
  } catch { /* library is optional */ }

  initViewer();
  refreshJobs();
}

function setModelStatus(info, err) {
  const dot = $('#aiDot');
  const txt = $('#aiText');
  if (!info) {
    dot.className = 'dot bad';
    txt.textContent = 'AI status unknown';
    txt.title = err || '';
    return;
  }
  const cfg = info.config || {};
  if (info.available) {
    dot.className = 'dot ok';
    txt.textContent = `AI: ${cfg.model || 'model'} ready`;
    txt.title = `Requests are planned by ${cfg.model} at ${cfg.base}.`;
  } else {
    dot.className = 'dot warn';
    txt.textContent = 'AI: keyword parser only';
    txt.title = 'No language model configured. Set CHATDFT_LLM_KEY or add '
      + 'data/llm.json; until then requests are parsed by built-in rules.';
  }
}

function setEngineStatus(health, err) {
  const dot = $('#engineDot');
  const txt = $('#engineText');
  if (!health) {
    dot.className = 'dot bad';
    txt.textContent = 'backend offline';
    txt.title = err || '';
    return;
  }
  const e = health.engines || {};
  if (e.pyscf && e.rdkit) {
    dot.className = 'dot ok';
    txt.textContent = `PySCF ${e.version || ''} ready`;
  } else if (e.pyscf) {
    dot.className = 'dot warn';
    txt.textContent = 'PySCF ready, RDKit missing';
  } else {
    dot.className = 'dot bad';
    txt.textContent = 'PySCF missing';
  }
}

function populateSelects() {
  const fsel = $('#selFunctional');
  const bsel = $('#selBasis');
  const prefF = ['b3lyp', 'pbe0', 'pbe', 'blyp', 'bp86', 'm06l', 'tpss', 'hf'];
  const prefB = ['6-31g*', '6-31g**', 'def2-svp', '6-31g', 'def2-tzvp', 'sto-3g'];
  const fmap = new Map(state.methods.functionals.map((f) => [f.key, f]));
  const bmap = new Map(state.methods.basis_sets.map((b) => [b.key, b]));

  const ordered = [
    ...prefF.filter((k) => fmap.has(k)),
    ...state.methods.functionals.map((f) => f.key).filter((k) => !prefF.includes(k)),
  ];
  fsel.innerHTML = ordered.map((k) =>
    `<option value="${k}">${escapeHtml(fmap.get(k).label)}</option>`).join('');
  fsel.value = 'b3lyp';

  const bOrdered = [
    ...prefB.filter((k) => bmap.has(k)),
    ...state.methods.basis_sets.map((b) => b.key).filter((k) => !prefB.includes(k)),
  ];
  bsel.innerHTML = bOrdered.map((k) =>
    `<option value="${k}">${escapeHtml(bmap.get(k).label)}</option>`).join('');
  bsel.value = '6-31g*';
}

/* ------------------------------------------------------------------ viewer */
function initViewer() {
  const node = $('#viewer');
  if (typeof $3Dmol === 'undefined') {
    node.innerHTML = '<div class="empty-state"><p>3D viewer could not load ' +
      '(no network access to the CDN). Calculations still work &mdash; use the ' +
      'Geometry tab for numeric structure data.</p></div>';
    return;
  }
  state.viewer = $3Dmol.createViewer(node, {
    backgroundColor: '#eef2f7',
    antialias: true,
  });
}

function viewerStyle() {
  switch (state.style) {
    case 'stick':     return { stick: { radius: 0.14 }, sphere: { scale: 0.24 } };
    case 'sphere':    return { sphere: { scale: 0.95 } };
    case 'line':      return { line: { linewidth: 2 } };
    default:          return { stick: { radius: 0.13 }, sphere: { scale: 0.28 } };
  }
}

function showMolecule(mol, opts) {
  opts = opts || {};
  state.molecule = mol;
  $('#viewerEmpty')?.classList.add('hidden');

  const meta = $('#molMeta');
  meta.textContent = `${mol.name || mol.formula} \u00b7 ${mol.formula} \u00b7 ` +
    `${mol.natoms} atoms \u00b7 ${mol.nelectrons} e\u207b`;

  if (!state.viewer) return;

  state.viewer.removeAllModels();
  const xyz = opts.xyz || molToXyz(mol);
  const model = state.viewer.addModel(xyz, 'xyz');
  state.viewer.setStyle({}, viewerStyle());

  if (opts.labels) {
    state.viewer.addPropertyLabels('atom', {}, {
      fontSize: 11,
      fontColor: '#16202e',
      showBackground: false,
      alignment: 'center',
    });
  }

  state.viewer.zoomTo();
  state.viewer.zoom(1.15);
  state.viewer.render();
  if (state.spin) state.viewer.spin('y');

  renderLegend(mol);
}

function molToXyz(mol) {
  let s = `${mol.natoms}\n${mol.name || mol.formula}\n`;
  for (const a of mol.atoms) {
    s += `${a.symbol} ${a.x.toFixed(8)} ${a.y.toFixed(8)} ${a.z.toFixed(8)}\n`;
  }
  return s;
}

function renderLegend(mol) {
  const box = $('#legend');
  const seen = new Map();
  for (const a of mol.atoms) {
    if (!seen.has(a.symbol)) {
      seen.set(a.symbol, (seen.get(a.symbol) || 0) + 1);
    } else {
      seen.set(a.symbol, seen.get(a.symbol) + 1);
    }
  }
  const colors = {
    H: '#FFFFFF', C: '#909090', N: '#3050F8', O: '#FF0D0D',
    F: '#90E050', S: '#FFFF30', Cl: '#1FF01F', Br: '#A62929',
    I: '#940094', P: '#FF8000', Na: '#AB5CF2', K: '#8F40D4',
    Fe: '#E06633', Cu: '#C88033', Zn: '#7D80B0',
  };
  box.innerHTML = Array.from(seen.entries()).map(([sym, n]) =>
    `<span class="item"><span class="swatch" style="background:${colors[sym] || '#cccccc'}"></span>` +
    `${sym}<span class="muted">&times;${n}</span></span>`
  ).join('');
}

/* ------------------------------------------------------------------ chat send */
async function sendMessage(text) {
  if (!text.trim()) return;

  addMessage('user', escapeHtml(text));
  $('#input').value = '';
  autoGrow($('#input'));

  const thinking = addThinking('Planning the calculation');

  const payload = {
    message: text,
    auto_run: true,
    functional: $('#selFunctional').value || null,
    basis: $('#selBasis').value || null,
  };
  const chg = parseInt($('#inpCharge').value, 10);
  if (!Number.isNaN(chg)) payload.charge = chg;
  const mult = parseInt($('#inpMult').value, 10);
  if (!Number.isNaN(mult)) payload.multiplicity = mult;

  try {
    const res = await api('/api/chat', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    removeThinking();

    let extras = null;
    if (res.intent && res.intent.molecule) {
      extras = renderIntentCard(res.intent, res.molecule, res.planner);
    }

    if (res.intent && res.intent.notes && res.intent.notes.length) {
      // Planner notes the user has to see: the model was not used, or the two
      // planners read the sentence as two different jobs and the model's
      // reading won.  A disagreement that is not shown here is a reply that
      // answers a different question from the one that was asked, with
      // nothing on screen to say so.  backend/check_planner.py reads this
      // regex out of the file rather than copying it.
      const note = res.intent.notes.filter((n) => /unavailable|fallback|disagreed/i.test(n));
      if (note.length) {
        const warn = el('div', 'warn-box');
        warn.textContent = note.join(' ');
        if (!extras) extras = warn; else extras.appendChild(warn);
      }
    }

    addMessage('assistant', markdown(res.reply || ''), extras);

    if (res.molecule) {
      const optXyz = res.result && res.result.optimized_xyz;
      showMolecule(res.molecule, { xyz: optXyz || undefined });
      renderMoleculeOnly(res.molecule);
    }

    // 'design' and 'series' arrive from the chat router the same way
    // 'compute' does; without them those requests would answer in words and
    // never start a job.
    if ((res.action === 'compute' || res.action === 'design' ||
         res.action === 'series' || res.action === 'field' ||
         res.action === 'reaction' || res.action === 'nto') && res.job_id) {
      state.activeJob = res.job_id;
      beginPolling(res.job_id);
    }

  } catch (e) {
    removeThinking();
    addMessage('assistant',
      `<div class="error-box">${escapeHtml(e.message)}</div>`);
  }
}

/* ------------------------------------------------------------------ polling */
function beginPolling(jobId) {
  if (state.pollTimer) clearInterval(state.pollTimer);
  showJobStatus('queued', 0, 'Submitted');

  const tick = async () => {
    let job;
    try {
      job = await api(`/api/job/${jobId}`);
    } catch (e) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      showJobStatus('failed', 100, e.message);
      return;
    }

    showJobStatus(job.status, job.percent, job.message);

    if (job.status === 'completed') {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      state.activeJob = null;
      handleCompleted(job);
    } else if (job.status === 'failed') {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      state.activeJob = null;
      addMessage('assistant',
        `<div class="error-box">Calculation failed:\n${escapeHtml(job.error || 'unknown error')}</div>`);
      refreshJobs();
    } else {
      refreshJobs(true);
    }
  };

  state.pollTimer = setInterval(tick, 700);
  tick();
}

function showJobStatus(status, percent, message) {
  const wrap = $('#jobStatus');
  const bar = $('#jobBar');
  const stage = $('#jobStage');
  if (!wrap) return;
  wrap.className = 'job-status ' + status;
  bar.style.width = Math.max(0, Math.min(100, percent || 0)) + '%';
  stage.textContent = message || status;
}

/* ------------------------------------------------------------------ results */
function handleCompleted(job) {
  const result = job.result;
  if (!result) return;

  refreshJobs();

  if (job.kind === 'compare') {
    renderCompare(job);
    return;
  }

  // a series is a set of molecules, not one; it gets its own panel
  if (job.kind === 'series') {
    renderSeries(job);
    return;
  }

  // a field map is a figure in a plane, not a set of numbers
  if (job.kind === 'field') {
    renderField(job);
    return;
  }

  // a reaction path is an energy profile with its own verdict badge
  if (job.kind === 'reaction') {
    renderReaction(job);
    return;
  }

  // a design run has no single molecule to narrate; it gets its own panel
  if (job.kind === 'design') {
    refreshJobs();
    renderDesign(job);
    return;
  }

  state.lastResult = result;

  // structure
  const xyz = result.optimized_xyz;
  if (result.molecule) {
    showMolecule(result.molecule, { xyz: xyz || undefined });
    afterOptimisation(result);
  }

  renderProperties(result);

  if (result.orbital_ladder && result.orbital_ladder.length) {
    renderOrbitals(result);
  }
  if (result.excited_states && result.excited_states.length) {
    renderSpectrum(result);
  }
  if (result.geometry) {
    renderGeometry(result.geometry);
  } else if (result.molecule && result.molecule.bonds) {
    renderGeometryFromBonds(result.molecule);
  }
  renderScan(result);
  renderVibrations(result);
  renderNmr(result);
  renderDos(result);
  renderReactivity(result);
  renderSurfaces(result);
  renderNCI(result);

  if (job.explanation) {
    appendNarrative(job, job.explanation);
    addMessage('assistant', markdown(job.explanation));
  }

  // warnings
  const warns = result.warnings || [];
  if (warns.length) {
    const box = el('div', 'warn-box');
    box.innerHTML = warns.map(escapeHtml).join('<br>');
    $('#propContent')?.appendChild(box);
  }

  // auto-switch to the tab holding the figure this job produced.  A scan used
  // to land on Properties: the curve was computed and drawn, and then the
  // user was shown a panel that had nothing to do with it.
  if (result.nci && result.nci.scatter) switchTab('nci');
  else if (result.surfaces && result.surfaces.surfaces &&
      result.surfaces.surfaces.length) switchTab('surfaces');
  else if (result.excited_states && result.excited_states.length) switchTab('spectrum');
  else if (job.kind === 'scan' && (result.scan || {}).points) switchTab('scan');
  else if (job.kind === 'vibrations' && result.ir_spectrum) switchTab('vibrations');
  else if (job.kind === 'nmr' && result.nuclei) switchTab('nmr');
  else if (job.kind === 'dos' && result.dos) switchTab('dos');
  else if (result.optimized_xyz) switchTab('geometry');
  else switchTab('properties');
}

function afterOptimisation(result) {
  if (!result.optimized_xyz) return;
  // realign the viewer with the relaxed geometry
  const lines = result.optimized_xyz.trim().split('\n');
  const n = parseInt(lines[0], 10);
  const atoms = [];
  for (let i = 2; i < 2 + n; i++) {
    const p = lines[i].trim().split(/\s+/);
    atoms.push({ symbol: p[0], x: +p[1], y: +p[2], z: +p[3] });
  }
  const mol = Object.assign({}, result.molecule, { atoms: atoms });
  showMolecule(mol, { xyz: result.optimized_xyz });
}

/* ---- properties -------------------------------------------------------- */
function renderProperties(r) {
  $('#emptyProps')?.classList.add('hidden');
  $('#propContent')?.classList.remove('hidden');

  const grid = $('#statGrid');
  grid.innerHTML = '';

  const stats = [
    { label: 'Total energy', value: fmt(r.energy_hartree, 6), unit: 'Ha', accent: true },
    { label: 'Energy', value: fmt(r.energy_ev, 2), unit: 'eV' },
    { label: 'HOMO', value: fmt(r.homo_ev, 3), unit: 'eV' },
    { label: 'LUMO', value: fmt(r.lumo_ev, 3), unit: 'eV' },
    { label: 'HOMO-LUMO gap', value: fmt(r.gap_ev, 3), unit: 'eV', accent: true },
    { label: 'Dipole', value: fmt((r.dipole || {}).magnitude, 3), unit: 'D' },
    { label: 'Electrons', value: String(r.nelec), unit: '' },
    { label: 'Basis functions', value: String(r.nbf), unit: '' },
    { label: 'SCF', value: r.converged ? 'converged' : 'not converged',
      unit: '', cls: r.converged ? '' : 'warn' },
    { label: 'Time', value: fmt(r.scf_seconds, 2), unit: 's' },
  ];

  // A solvated calculation and a gas-phase one are different numbers with
  // the same name, so the model has to be readable off the result -- the
  // request that asked for it is not part of the figure.
  if (r.solvation) {
    stats.push({
      label: 'Solvent',
      value: `\u03B5 = ${String(r.solvation)}`,
      unit: r.solvation_model || 'implicit',
    });
  }

  // An unrestricted determinant is not an eigenfunction of S^2, so an
  // open-shell result that does not report <S^2> cannot be told apart from
  // one that converged to the wrong spin state.  Show the exact value it
  // should have had next to it.
  if (r.spin && r.spin.s_squared !== null && r.spin.s_squared !== undefined) {
    const sp = r.spin;
    stats.push({
      label: '\u27E8S\u00B2\u27E9',
      value: fmt(sp.s_squared, 4),
      unit: ` vs ${fmt(sp.s_squared_expected, 3)} exact`,
      cls: Math.abs(sp.contamination_pct || 0) > 10 ? 'warn' : '',
    });
  }

  // A relaxation that did not reach the gradient criterion is not a minimum,
  // and everything computed on it inherits that.  Say which, and show the
  // residual gradient the verdict was read off.
  if (r.optimized_xyz) {
    const g = (r.opt_grms === null || r.opt_grms === undefined)
      ? null : Number(r.opt_grms);
    stats.push({
      label: 'Geometry',
      value: r.opt_converged === false ? 'not converged'
        : (r.opt_converged ? 'stationary point' : 'not checked'),
      unit: g === null ? '' : ` |g|rms ${g.toExponential(1)}`,
      cls: r.opt_converged === false ? 'warn' : '',
    });
  }

  for (const s of stats) {
    // ``cls`` was declared on the SCF chip and never applied, so a
    // non-converged SCF looked exactly like a converged one.
    const node = el('div', 'stat' + (s.accent ? ' accent' : '')
      + (s.cls ? ' ' + s.cls : ''));
    node.innerHTML =
      `<div class="label">${s.label}</div>` +
      `<div class="value">${s.value}<span class="unit">${s.unit || ''}</span></div>`;
    grid.appendChild(node);
  }

  // dipole
  const d = r.dipole || {};
  const dblock = $('#dipoleBlock');
  const mag = d.magnitude || 0;
  const width = Math.min(100, Math.max(2, (mag / 6) * 100));
  dblock.innerHTML =
    `<div class="dipole-value">${fmt(mag, 3)}<span class="unit"> D</span></div>` +
    `<div class="dipole-vec">` +
      `\u03bc<sub>x</sub> = ${fmt(d.x, 3)} D<br>` +
      `\u03bc<sub>y</sub> = ${fmt(d.y, 3)} D<br>` +
      `\u03bc<sub>z</sub> = ${fmt(d.z, 3)} D</div>` +
    `<div class="dipole-arrow" style="width:${width}%"></div>`;

  // charges
  const charges = r.mulliken_charges || [];
  const lowdin = r.lowdin_charges || [];
  const bonding = r.bonding || {};
  const hirshfeld = (bonding.hirshfeld && bonding.hirshfeld.charges) || [];
  const loByAtom = new Map(lowdin.map((c) => [c.atom, c]));
  const hiByAtom = new Map(hirshfeld.map((c) => [c.atom, c]));
  const cblock = $('#chargeBlock');
  cblock.innerHTML = '';
  if (!charges.length) {
    cblock.innerHTML = '<p class="muted">No population analysis available.</p>';
  } else {
    const maxAbs = Math.max(
      0.1,
      ...charges.map((c) => Math.abs(c.charge)),
      ...lowdin.map((c) => Math.abs(c.charge)),
      ...hirshfeld.map((c) => Math.abs(c.charge)),
    );
    const bar = (q, extra) => {
      const pct = (Math.abs(q) / maxAbs) * 50;
      const pos = q >= 0;
      return `<span class="charge-bar${extra}"><span class="mid"></span>` +
        `<span class="${pos ? 'pos' : 'neg'}" style="width:${pct}%"></span></span>`;
    };
    const num = (q) => `${q >= 0 ? '+' : ''}${q.toFixed(3)}`;
    charges.forEach((c) => {
      const l = loByAtom.get(c.atom);
      const h = hiByAtom.get(c.atom);
      const row = el('div', 'charge-row');
      row.innerHTML =
        `<span class="atom">${c.symbol}${c.atom}</span>` +
        bar(c.charge, '') +
        `<span class="val">${num(c.charge)}</span>` +
        (l ? bar(l.charge, ' lo') + `<span class="val">${num(l.charge)}</span>` : '') +
        (h ? bar(h.charge, ' hi') + `<span class="val">${num(h.charge)}</span>` : '');
      cblock.appendChild(row);
    });
  }

  // The three partitions disagree by design -- Mulliken splits overlaps
  // evenly, Hirshfeld weights them by the free-atom densities.  Showing the
  // Hirshfeld charges without saying whether they are trustworthy would be
  // worse than not showing them: the grid integral is the number that says
  // the density was evaluated correctly at all.
  const note = $('#hirshfeldNote');
  const hd = (bonding.hirshfeld && bonding.hirshfeld.diagnostics) || {};
  if (note) {
    if (hd.grid_integral_electrons !== undefined) {
      note.classList.remove('hidden');
      note.textContent =
        `Hirshfeld: charges sum to ${fmt(hd.charge_sum, 5)} e (the molecule ` +
        `carries ${fmt(hd.expected_charge_sum, 5)}); the density integrates to ` +
        `${fmt(hd.grid_integral_electrons, 5)} of ${hd.n_electrons} electrons ` +
        `(${fmt(hd.grid_integral_error_pct, 4)}%). Free atoms are spherically ` +
        `averaged, so equivalent atoms come out equivalent.`;
    } else {
      note.classList.add('hidden');
    }
  }

  // bonds
  const mayer = bonding.mayer || {};
  const bonds = mayer.bonds || [];
  // Deliberately not `bondTable`: the Geometry pane already owns that id for
  // the perceived bond list, and a duplicate would have made this render into
  // whichever element came first in the document.
  const btable = $('#mayerTable');
  const bnote = $('#mayerNote');
  if (btable) {
    if (!bonds.length) {
      btable.innerHTML = '';
      if (bnote) {
        bnote.classList.remove('hidden');
        bnote.textContent = bonding.mayer_error
          ? `Bond orders unavailable: ${bonding.mayer_error}`
          : `No bond orders above the reporting threshold ` +
            `(${mayer.threshold}); this molecule has no bonds by that measure.`;
      }
    } else {
      btable.innerHTML =
        '<thead><tr><th>Bond</th><th>Mayer order</th><th></th></tr></thead>' +
        '<tbody>' + bonds.map((b) => {
          const pct = Math.min(100, (b.order / Math.max(1, bonds[0].order)) * 100);
          return `<tr><td>${b.from}${b.i} &ndash; ${b.to}${b.j}</td>` +
            `<td>${b.order.toFixed(3)}</td>` +
            `<td><span class="order-bar" style="width:${pct}%"></span></td></tr>`;
        }).join('') + '</tbody>';
      if (bnote) {
        bnote.classList.remove('hidden');
        const v = (mayer.valence || [])
          .map((x) => `${x.symbol}${x.atom} ${x.valence.toFixed(2)}`).join(', ');
        bnote.textContent =
          `Total bond order ${fmt(mayer.total_bond_order, 3)}; Mayer valence: ` +
          `${v}. ${mayer.convention || ''}`;
      }
    }
  }
}

/* ---- orbitals ---------------------------------------------------------- */
function renderOrbitals(r) {
  $('#orbContent')?.classList.remove('hidden');
  const p = document.querySelector('[data-pane="orbitals"] .empty-state');
  if (p) p.classList.add('hidden');

  const ladder = r.orbital_ladder;
  const labels = ladder.map((o) => {
    if (o.label) return o.label + ' (' + (o.index + 1) + ')';
    return '#' + (o.index + 1);
  });
  const values = ladder.map((o) => o.energy_ev);
  const nocc = r.nocc;

  const colors = ladder.map((o, i) =>
    i < nocc ? 'rgba(31,111,235,.85)' : 'rgba(210,105,30,.75)');

  const ctx = $('#orbChart').getContext('2d');
  if (state.charts.orb) state.charts.orb.destroy();
  state.charts.orb = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: colors.map((c) => c.replace(/[\d.]+\)$/, '1)')),
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: `Kohn-Sham orbital energies, ${r.ladder_channel || 'closed shell'}`
            + ' channel (blue = occupied, orange = virtual)',
          color: '#46556b',
          font: { size: 11, weight: 'normal' },
        },
        tooltip: {
          callbacks: {
            label: (c) => `${c.parsed.y.toFixed(3)} eV`,
          },
        },
      },
      scales: {
        y: {
          title: { display: true, text: 'Energy (eV)', color: '#7a8797', font: { size: 11 } },
          ticks: { color: '#7a8797', font: { size: 10 } },
          grid: { color: '#e4e9f0' },
        },
        x: {
          ticks: { color: '#7a8797', font: { size: 9 }, maxRotation: 90, minRotation: 60 },
          grid: { display: false },
        },
      },
    },
  });

  // frontier composition
  const fb = $('#frontierBlock');
  const fo = r.frontier_orbitals || {};
  const pal = ['#1f6feb', '#d2691e', '#16a34a', '#9333ea', '#b45309', '#0891b2', '#be185d'];
  fb.innerHTML = '';
  [['homo', 'HOMO'], ['lumo', 'LUMO']].forEach(([key, title]) => {
    // The payload lists atoms in index order, not by weight, so sort before
    // picking the dominant one -- otherwise the label contradicts the bars.
    const comp = (fo[key] || []).slice().sort((a, b) => b.percent - a.percent);
    if (!comp.length) return;
    const row = el('div', 'frontier-row');
    const bars = comp.slice(0, 8).map((c, i) =>
      `<span style="width:${c.percent}%;background:${pal[i % pal.length]}" ` +
      `title="${c.symbol}${c.atom}: ${c.percent}%"></span>`).join('');
    row.innerHTML =
      `<div class="head"><span class="name">${title}</span>` +
      `<span class="pct">dominant: ${comp[0].symbol}${comp[0].atom} ` +
      `${comp[0].percent.toFixed(1)}%</span></div>` +
      `<div class="frontier-bar">${bars}</div>`;
    fb.appendChild(row);
  });
  if (!fb.children.length) fb.innerHTML = '<p class="muted">Composition not available for this system.</p>';

  // table
  const table = $('#orbTable');
  let html = '<thead><tr><th>#</th><th>Energy (Ha)</th><th>Energy (eV)</th>' +
    '<th>Occ.</th><th>Label</th></tr></thead><tbody>';
  ladder.forEach((o) => {
    const hl = o.label === 'HOMO' || o.label === 'LUMO' ? ' class="highlight"' : '';
    html += `<tr${hl}><td>${o.index + 1}</td><td>${o.energy_ha.toFixed(5)}</td>` +
      `<td>${o.energy_ev.toFixed(3)}</td><td>${o.occupation.toFixed(1)}</td>` +
      `<td>${o.label || ''}</td></tr>`;
  });
  table.innerHTML = html + '</tbody>';
}

/* ---- spectrum ---------------------------------------------------------- */
/* A UV-Vis figure has to say what produced it: TDA and full TD-DFT are
   different approximations and differ by tenths of an eV, so a caption that
   only says "TD-DFT" cannot be reproduced. */
function spectrumMethod(r) {
  const lvl = [r.functional_label || r.functional, r.basis_label || r.basis]
    .filter(Boolean).join('/');
  const m = r.excited_method_label || 'TD-DFT (Tamm-Dancoff)';
  return lvl ? `${m} at ${lvl}` : m;
}

/* The broadened absorption curve -- the figure a paper actually prints.
   The server has always built it ("the state list alone is not a figure"),
   and for a long time nothing drew it: the pane promised "the absorption
   curve" while showing only bars of oscillator strength.

   Two scales, because a paper needs both: the curve normalised to its own
   maximum (what a talk slide wants) and the absolute molar absorptivity in
   L mol-1 cm-1 (what a table quotes).  The shape is identical -- the second
   is the first times a constant -- so switching only rescales the y axis. */
function renderUvCurve(r) {
  const wrap = $('#uvWrap');
  const uv = r.uv_spectrum;
  if (!wrap) return;
  if (!uv || !(uv.x || []).length || !(uv.y || []).length) {
    wrap.classList.add('hidden');
    return;
  }
  wrap.classList.remove('hidden');
  const absolute = state.uvScale === 'epsilon' && (uv.epsilon || []).length;
  const data = absolute ? uv.epsilon : uv.y;
  const ytitle = absolute
    ? (uv.epsilon_ylabel || 'Molar absorptivity (L mol-1 cm-1)')
    : (uv.ylabel || 'Relative absorbance (%)');
  const ctx = $('#uvChart').getContext('2d');
  if (state.charts.uv) state.charts.uv.destroy();
  const fwhm = uv.fwhm_ev !== undefined && uv.fwhm_ev !== null
    ? `  —  Gaussian broadening, FWHM ${uv.fwhm_ev} eV` : '';
  state.charts.uv = new Chart(ctx, {
    type: 'line',
    data: {
      labels: uv.x,
      datasets: [{
        label: ytitle,
        data: data,
        borderColor: 'rgba(31,111,235,1)',
        backgroundColor: 'rgba(31,111,235,.12)',
        borderWidth: 1.6,
        pointRadius: 0,
        fill: true,
        tension: 0.15,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: `${spectrumMethod(r)} simulated absorption${fwhm}`,
          color: '#46556b',
          font: { size: 11, weight: 'normal' },
        },
        tooltip: {
          callbacks: {
            title: (c) => `${c[0].label} nm`,
            label: (c) => absolute
              ? `\u03b5 = ${c.parsed.y.toPrecision(4)} L mol-1 cm-1`
              : `${ytitle} ${c.parsed.y.toFixed(2)}`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: uv.xlabel || 'Wavelength (nm)', color: '#7a8797', font: { size: 11 } },
          ticks: { color: '#7a8797', font: { size: 10 }, maxTicksLimit: 10 },
          grid: { display: false },
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: ytitle, color: '#7a8797', font: { size: 11 } },
          ticks: { color: '#7a8797', font: { size: 10 } },
          grid: { color: '#e4e9f0' },
        },
      },
    },
  });
  renderUvNote(uv);
}

/* What the figure is actually showing, and what it is not.  The window is
   reported because it is now taken from the states rather than fixed at
   180-800 nm, where a small molecule's bands fall outside it and the plot
   shows nothing but Gaussian tails normalised to 100%.  The sum rule is
   reported because it is the one number that says whether the broadening and
   the absolute scale are right -- and because sum(f)/N_electrons is how much
   of the spectrum these roots captured, which a stick list cannot say. */
function renderUvNote(uv) {
  const el = $('#uvNote');
  if (!el) return;
  const num = (v, d) => (v === null || v === undefined || Number.isNaN(v))
    ? '-' : Number(v).toFixed(d);
  const fmtInt = (v) => (v === null || v === undefined || Number.isNaN(v)) ? '-'
    : (Math.abs(v) >= 1e5 ? Number(v).toExponential(3) : Number(v).toFixed(1));
  const bits = [];
  const win = uv.window_nm || [];
  if (win.length === 2) {
    const how = uv.window_source === 'auto' ? 'from the states'
      : uv.window_source === 'explicit' ? 'as requested' : 'fallback';
    bits.push(`Window ${win[0]}&ndash;${win[1]} nm (${how}).`);
  }
  const st = uv.strongest || {};
  if (st.oscillator_strength !== undefined) {
    const eps = (uv.peaks || []).find((p) => p.state === st.state);
    const emax = eps && eps.epsilon_max_l_mol_cm !== undefined
      ? `, \u03b5<sub>max</sub> ${fmtInt(eps.epsilon_max_l_mol_cm)} L mol<sup>-1</sup> cm<sup>-1</sup>`
      : '';
    bits.push(`Strongest band S${st.state} at ${num(st.wavelength_nm, 1)} nm`
      + ` (f = ${num(st.oscillator_strength, 4)}${emax}).`);
  }
  const sr = uv.sum_rule || {};
  if (sr.residual_pct !== undefined) {
    const ok = Math.abs(sr.residual_pct) < 0.5;
    bits.push(`\u222b\u03b5 d\u03bd = ${fmtInt(sr.measured_l_mol_cm2)} vs`
      + ` ${fmtInt(sr.expected_l_mol_cm2)} L mol<sup>-1</sup> cm<sup>-2</sup>`
      + ` expected from \u03a3f (${ok ? 'agrees' : 'disagrees'} to`
      + ` ${Math.abs(sr.residual_pct).toFixed(3)}%).`);
  }
  if (sr.sum_f_over_n_electrons !== undefined && sr.sum_f_over_n_electrons !== null) {
    bits.push(`\u03a3f = ${num(sr.sum_f, 4)} over ${sr.n_used} of`
      + ` ${sr.n_states} roots, ${(100 * sr.sum_f_over_n_electrons).toFixed(1)}%`
      + ` of the Thomas\u2013Reiche\u2013Kuhn sum rule (${sr.n_electrons} electrons)`
      + ` \u2014 the rest of the spectrum lies above these roots.`);
  }
  let html = bits.join(' ');
  (uv.notes || []).forEach((n) => { html += ` <strong class="warn">${n}</strong>`; });
  el.innerHTML = html;
  el.classList.toggle('hidden', !html);
}

function renderSpectrum(r) {
  $('#specContent')?.classList.remove('hidden');
  const p = document.querySelector('[data-pane="spectrum"] .empty-state');
  if (p) p.classList.add('hidden');

  renderUvCurve(r);

  const states = r.excited_states;
  const labels = states.map((s) => `S${s.state}`);
  const energies = states.map((s) => s.energy_ev);
  const strengths = states.map((s) => s.oscillator_strength || 0);
  const maxF = Math.max(0.01, ...strengths);

  const ctx = $('#specChart').getContext('2d');
  if (state.charts.spec) state.charts.spec.destroy();
  state.charts.spec = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Oscillator strength',
        data: strengths,
        backgroundColor: 'rgba(31,111,235,.8)',
        borderColor: 'rgba(31,111,235,1)',
        borderWidth: 1,
        yAxisID: 'y',
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: `${spectrumMethod(r)} vertical excitations `
                + '(bar height = oscillator strength)',
          color: '#46556b',
          font: { size: 11, weight: 'normal' },
        },
        tooltip: {
          callbacks: {
            label: (c) => {
              const s = states[c.dataIndex];
              return [
                `${s.energy_ev.toFixed(3)} eV  (${s.wavelength_nm ? s.wavelength_nm.toFixed(1) : '-'} nm)`,
                `f = ${(s.oscillator_strength || 0).toFixed(4)}`,
              ];
            },
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          suggestedMax: maxF * 1.2,
          title: { display: true, text: 'Oscillator strength f', color: '#7a8797', font: { size: 11 } },
          ticks: { color: '#7a8797', font: { size: 10 } },
          grid: { color: '#e4e9f0' },
        },
        x: {
          title: { display: true, text: 'Excited state', color: '#7a8797', font: { size: 11 } },
          ticks: { color: '#7a8797', font: { size: 10 } },
          grid: { display: false },
        },
      },
    },
  });

  const table = $('#specTable');
  // eps_max comes from the server's broadened curve: it is the analytic
  // maximum of an isolated band, so it does not depend on where the plot
  // window was cut -- which is what makes it quotable in a table.
  const epsOf = {};
  ((r.uv_spectrum || {}).peaks || []).forEach((p) => { epsOf[p.state] = p; });
  const epsNum = (v) => (v === null || v === undefined) ? '-'
    : (v >= 1e5 ? v.toExponential(3) : v.toFixed(0));
  let html = '<thead><tr><th>State</th><th>Energy (eV)</th><th>Wavelength (nm)</th>' +
    '<th>f</th><th>&epsilon;<sub>max</sub> (L mol<sup>-1</sup> cm<sup>-1</sup>)</th>' +
    '<th>Optically active</th></tr></thead><tbody>';
  states.forEach((s) => {
    const hl = s.active ? ' class="highlight"' : '';
    const ep = epsOf[s.state] || {};
    html += `<tr${hl}><td>S${s.state}</td><td>${s.energy_ev.toFixed(3)}</td>` +
      `<td>${s.wavelength_nm ? s.wavelength_nm.toFixed(1) : '-'}</td>` +
      `<td>${s.oscillator_strength !== undefined ? s.oscillator_strength.toFixed(4) : '-'}</td>` +
      `<td>${epsNum(ep.epsilon_max_l_mol_cm)}</td>` +
      `<td>${s.active ? 'yes' : 'no'}</td></tr>`;
  });
  table.innerHTML = html + '</tbody>';
}

/* ---- bond scan -------------------------------------------------------- */
function renderScan(r) {
  const sc = r.scan;
  const content = $('#scanContent');
  const empty = document.querySelector('[data-pane="scan"] .empty-state');
  if (!sc || !sc.points || !sc.points.length) {
    content?.classList.add('hidden');
    empty?.classList.remove('hidden');
    return;
  }
  content?.classList.remove('hidden');
  empty?.classList.add('hidden');

  const pts = sc.points;
  const pair = (sc.labels || []).join('-') || sc.element_pair || 'bond';

  const ctx = $('#scanChart').getContext('2d');
  if (state.charts.scan) state.charts.scan.destroy();
  state.charts.scan = new Chart(ctx, {
    type: 'line',
    data: {
      labels: pts.map((p) => p.r.toFixed(3)),
      datasets: [{
        label: 'E - Emin (kcal/mol)',
        data: pts.map((p) => p.energy_rel_kcal),
        borderColor: 'rgba(31,111,235,1)',
        backgroundColor: 'rgba(31,111,235,.15)',
        borderWidth: 2,
        pointRadius: 3,
        fill: true,
        tension: 0.25,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: `bond length (A)` } },
        y: {
          title: { display: true, text: 'E - Emin (kcal/mol)' },
          beginAtZero: true,
        },
      },
      plugins: {
        // Read the word off the payload instead of asserting it: a relaxed
        // scan would otherwise still be captioned "rigid", and quoting its
        // minimum as an equilibrium bond length is exactly the error that
        // word is there to prevent.
        title: {
          display: true,
          text: `${sc.rigid ? 'Rigid' : 'Relaxed'} scan: ${pair} bond`,
        },
        legend: { display: false },
      },
    },
  });

  const table = $('#scanTable');
  let html = '<thead><tr><th>#</th><th>r (A)</th><th>E (Ha)</th>' +
    '<th>E - Emin (kcal/mol)</th><th>Converged</th></tr></thead><tbody>';
  pts.forEach((p, i) => {
    const hl = i === sc.minimum_index ? ' class="highlight"' : '';
    html += `<tr${hl}><td>${i + 1}</td><td>${p.r.toFixed(3)}</td>` +
      `<td>${p.energy_hartree.toFixed(6)}</td>` +
      `<td>${p.energy_rel_kcal.toFixed(2)}</td>` +
      `<td>${p.converged ? 'yes' : 'no'}</td></tr>`;
  });
  table.innerHTML = html + '</tbody>';

  // The lowest sampled point is only a grid point.  Refining the grid does
  // not steadily improve it (on water's O-H well it is off by 0.004 A at 12
  // points and still 0.009 A at 25), so showing it alone invites the reader
  // to quote a bond length whose last digit is noise.
  const note = $('#scanNote');
  if (note) {
    const fit = sc.minimum_fitted;
    const step = Number(sc.grid_step) || 0;
    if (fit) {
      note.innerHTML =
        `Minimum (parabola vertex): <b>${fit.r.toFixed(4)} \u00C5</b> \u00B7 ` +
        `lowest sampled point ${sc.minimum.r.toFixed(3)} \u00C5 \u00B7 ` +
        `grid step ${step.toFixed(3)} \u00C5. Rigid scan \u2014 the true ` +
        `equilibrium length needs a relaxed optimisation.`;
    } else {
      note.innerHTML =
        `Lowest sampled point: <b>${sc.minimum.r.toFixed(3)} \u00C5</b> ` +
        `\u00B7 grid step ${step.toFixed(3)} \u00C5, so treat it as uncertain ` +
        `by up to one grid step.`;
    }
    note.classList.remove('hidden');
  }
}

/* ---- geometry ---------------------------------------------------------- */
function renderVibrations(r) {
  const v = r.ir_spectrum;
  const box = $('#vibContent');
  if (!box || !v || !v.x || !v.x.length) return;
  box.classList.remove('hidden');
  const p = document.querySelector('[data-pane="vibrations"] .empty-state');
  if (p) p.classList.add('hidden');

  const ctx = $('#vibChart').getContext('2d');
  if (state.charts.vib) state.charts.vib.destroy();
  state.charts.vib = new Chart(ctx, {
    type: 'line',
    data: {
      labels: v.x,
      datasets: [{
        label: 'IR intensity',
        data: v.y,
        borderColor: 'rgba(31,111,235,1)',
        backgroundColor: 'rgba(31,111,235,.15)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          // the payload already runs high -> low, the IR convention
          title: { display: true, text: v.xlabel || 'Wavenumber (cm-1)' },
        },
        y: {
          title: { display: true, text: v.ylabel || 'Relative intensity (%)' },
          beginAtZero: true,
        },
      },
    },
  });

  const th = r.thermochemistry || {};
  const rows = [];
  if (r.zpe_kj_mol !== undefined) rows.push(['ZPE', r.zpe_kj_mol + ' kJ/mol']);
  if (th.entropy_j_mol_k !== undefined) rows.push(['S°(298 K)', th.entropy_j_mol_k + ' J/(mol K)']);
  if (th.heat_capacity_cv_j_mol_k !== undefined) rows.push(['Cv', th.heat_capacity_cv_j_mol_k + ' J/(mol K)']);
  if (th.gibbs_hartree !== undefined) rows.push(['G(298 K)', th.gibbs_hartree + ' Ha']);
  if (th.symmetry_number !== undefined) rows.push(['Rotational symmetry number', th.symmetry_number]);
  $('#vibThermo').innerHTML = rows.length
    ? rows.map(([k, val]) => `<span class="chip"><b>${k}</b> ${val}</span>`).join('')
    : '';
  if (r.n_imaginary) {
    $('#vibThermo').innerHTML +=
      `<span class="chip warn"><b>${r.n_imaginary} imaginary mode(s)</b> ` +
      `not a minimum: ${(r.imaginary_cm1 || []).join(', ')} cm-1</span>`;
  }

  // The scale factor is fitted per method and basis, so it belongs on the
  // figure: 0.899 at HF/6-31G* against 0.960 at B3LYP/6-31G* is a 250 cm-1
  // difference on an O-H stretch, which no reader can infer from a bare
  // "Scaled" column.
  const sf = Number(v.scale_factor);
  const scaled = Number.isFinite(sf) && Math.abs(sf - 1) > 1e-9;
  const chip = document.createElement('span');
  chip.className = scaled ? 'chip' : 'chip warn';
  chip.innerHTML = scaled
    ? `<b>scaled \u00D7${sf.toFixed(3)}</b> ${escapeHtml(v.scale_source || '')}`
    : `<b>unscaled</b> ${escapeHtml(v.scale_source || 'no published factor for this level')}`;
  $('#vibThermo').appendChild(chip);

  // Whether the km/mol column means anything at all.  The plotted curve is
  // normalised to 100, so a spectrum whose every intensity is 4x too large
  // looks identical to a correct one; the intensity sum rule is the only
  // thing that separates them, so report it next to the numbers.
  const sr = r.ir_sum_rule || {};
  if (sr.residual_pct !== undefined) {
    const abs = Math.abs(Number(sr.residual_pct)) <= 0.5;
    const c2 = document.createElement('span');
    c2.className = abs ? 'chip' : 'chip warn';
    c2.innerHTML = abs
      ? `<b>absolute intensities</b> sum rule closes to ${sr.residual_pct}%`
      : `<b>intensities off absolute scale</b> sum rule residual ${sr.residual_pct}%`;
    c2.title = escapeHtml(r.ir_intensity_units || '');
    $('#vibThermo').appendChild(c2);
  }

  const peaks = v.peaks || [];
  $('#vibTable').innerHTML =
    '<thead><tr><th>Harmonic (cm-1)</th>' +
    (scaled ? '<th>Scaled (cm-1)</th>' : '') +
    '<th>IR intensity (km/mol)</th></tr></thead><tbody>' +
    peaks.map((q) => `<tr><td>${q.frequency_cm1}</td>` +
      (scaled ? `<td>${q.scaled_cm1}</td>` : '') +
      `<td>${q.intensity_km_mol}</td></tr>`).join('') +
    '</tbody>';

  renderRaman(r, scaled);
}

/* ---- NMR --------------------------------------------------------------- */
// The x axis is reversed, because an NMR spectrum is always plotted with high
// ppm on the left.  Drawing it the other way round produces a figure that
// looks right to someone who has never read one and is instantly wrong to
// anyone who has.
//
// One isotope at a time.  A 1H window is about 12 ppm wide and a 13C window
// about 220, so a single axis carrying both is not a spectrum anybody can
// compare with a measurement -- and the ppm width of a line depends on the
// Larmor frequency of the nucleus being observed, so the two do not even share
// a linewidth.  The engine returns one spectrum per isotope and the buttons
// below the title pick which one is drawn.
//
// The sticks are drawn from `spectrum.sticks`, which carries the integral of
// every group.  The caption used to promise them while the chart only ever
// drew the envelope.
const nmrSticks = {
  id: 'nmrSticks',
  afterDatasetsDraw(chart) {
    const sticks = chart.$nmrSticks;
    const area = chart.chartArea;
    if (!sticks || !sticks.length || !area) return;
    const xs = chart.data.labels;
    const sx = chart.scales.x;
    const sy = chart.scales.y;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = 'rgba(178,54,54,.9)';
    ctx.lineWidth = 1.3;
    sticks.forEach((s) => {
      // the nearest grid point, because the x scale is categorical
      let best = 0;
      let bestd = Infinity;
      for (let i = 0; i < xs.length; i += 1) {
        const d = Math.abs(Number(xs[i]) - Number(s.delta_ppm));
        if (d < bestd) { bestd = d; best = i; }
      }
      const px = sx.getPixelForValue(best);
      const py = sy.getPixelForValue(Number(s.rel_intensity) || 0);
      ctx.beginPath();
      ctx.moveTo(px, sy.getPixelForValue(0));
      ctx.lineTo(px, py);
      ctx.stroke();
    });
    ctx.restore();
  },
};

let nmrPayload = null;
let nmrIsotope = null;

function renderNmr(r) {
  const content = $('#nmrContent');
  const level = $('#nmrLevel');
  if (!content || !level) return;
  const nuclei = r.nuclei || [];
  if (!nuclei.length) {
    content.classList.add('hidden');
    return;
  }
  content.classList.remove('hidden');
  nmrPayload = r;

  // ---- level of theory, first, because it is the thing that gets misread --
  const lv = r.level || {};
  const chips = [];
  chips.push(`<span class="chip"><b>${escapeHtml(lv.scf || 'RHF')}</b> / `
    + `${escapeHtml(lv.basis || '')}</span>`);
  // The shielding never uses the requested functional.  Say it here, on the
  // figure, and not only in the prose: a reader who assumes B3LYP will
  // misread every absolute shielding.
  if (lv.functional_requested) {
    chips.push('<span class="chip warn"><b>Hartree-Fock shielding</b> '
      + `the requested ${escapeHtml(lv.functional_requested)} is not used`
      + '</span>');
  }
  const dg = r.diagnostics || {};
  const dE = (dg.dE_dB || []).map(Number);
  if (dE.length) {
    const worst = Math.max.apply(null, dE.map(Math.abs));
    const ok = worst < 1e-6;
    chips.push(ok
      ? `<span class="chip"><b>dE/dB = 0</b> consistency check passed</span>`
      : `<span class="chip warn"><b>dE/dB = ${worst.toExponential(1)}</b> `
        + 'a closed shell must have none</span>');
  }
  if (dg.tensor_asymmetry_max_ppm !== undefined) {
    const a = Number(dg.tensor_asymmetry_max_ppm);
    chips.push(`<span class="chip"><b>tensor symmetrised</b> `
      + `antisymmetric part ${a.toFixed(3)} ppm</span>`);
  }
  const refs = r.references || {};
  Object.keys(refs).sort().forEach((el) => {
    const q = refs[el];
    chips.push(`<span class="chip"><b>${escapeHtml(el)} ref</b> `
      + `${escapeHtml(q.label)} ${Number(q.sigma_iso_ppm).toFixed(2)} ppm`
      + (q.cached ? ' (cached)' : '') + '</span>');
    // A reference compound's name is not a scale.  For 15N the two scales in
    // common use are 380 ppm apart, so a shift quoted against NH3 and
    // compared with a paper that used nitromethane reads as a 380 ppm error
    // in the calculation unless this is said next to the number.
    if (q.scale_note) {
      chips.push(`<span class="chip warn"><b>${escapeHtml(el)} scale</b> `
        + `${escapeHtml(q.scale_note)}</span>`);
    }
    // Which scale a number is on is a different question from whether the
    // method reproduces it. A reference compound makes a shift definable, not
    // correct: this build puts 15N 24% high and 19F 22% low, and for 77Se the
    // method does not produce a shift at all. The note carries the measured
    // numbers, so a user comparing with a paper knows what to expect before
    // concluding the calculation is wrong.
    if (q.accuracy_note) {
      chips.push(`<span class="chip warn"><b>${el} accuracy</b> `
        + `${escapeHtml(q.accuracy_note)}</span>`);
    }
  });
  // An element with no reference has no shift, and the old payload simply
  // left it out of the spectrum without saying so.
  const noRef = r.unreferenced_elements || [];
  if (noRef.length) {
    chips.push('<span class="chip warn"><b>no reference</b> '
      + `${escapeHtml(noRef.join(', '))} has an absolute shielding but no `
      + 'chemical shift</span>');
  }
  // A reference that was computed and still gives no shift.  A separate chip
  // because the cause is different and so is the remedy: an unreferenced
  // nucleus needs a reference compound, this one needs a different level of
  // theory, and telling a user "no reference" when one was computed would
  // send them looking for the wrong thing.
  const noShift = r.scale_unusable_elements || [];
  if (noShift.length) {
    chips.push('<span class="chip warn"><b>scale not reproduced</b> '
      + `${escapeHtml(noShift.join(', '))} has a reference compound that was `
      + 'computed, but this method does not reproduce that nucleus\u2019s '
      + 'scale, so no chemical shift is reported</span>');
  }
  level.innerHTML = chips.join('');

  // ---- which isotope is on the axis -------------------------------------
  const spectra = r.spectra
    || (r.spectrum && r.spectrum_isotope
      ? { [r.spectrum_isotope]: r.spectrum } : null)
    || (r.spectrum && r.spectrum.shift_ppm && r.spectrum.shift_ppm.length
      ? { all: r.spectrum } : {});
  const keys = Object.keys(spectra).filter(
    (k) => spectra[k] && (spectra[k].shift_ppm || []).length);
  const seg = $('#nmrIsotope');
  if (seg) {
    if (keys.length > 1) {
      if (!nmrIsotope || !spectra[nmrIsotope]) {
        nmrIsotope = r.spectrum_isotope && spectra[r.spectrum_isotope]
          ? r.spectrum_isotope : keys[0];
      }
      seg.innerHTML = keys.map((k) => '<button type="button" '
        + `class="btn tiny ghost${k === nmrIsotope ? ' active' : ''}" `
        + `data-nmr-isotope="${escapeHtml(k)}">${escapeHtml(k)}</button>`)
        .join('');
      seg.classList.remove('hidden');
    } else {
      nmrIsotope = keys[0] || null;
      seg.innerHTML = keys.length
        ? `<span class="muted">${escapeHtml(keys[0])} only</span>` : '';
      seg.classList.remove('hidden');
    }
  }
  const spec = (nmrIsotope && spectra[nmrIsotope])
    ? spectra[nmrIsotope] : (keys.length ? spectra[keys[0]] : {});

  // ---- the spectrum ------------------------------------------------------
  const xs = (spec.shift_ppm || []).map(Number);
  const ys = (spec.intensity || []).map(Number);
  if (xs.length && ys.length) {
    const ctx = $('#nmrChart').getContext('2d');
    if (state.charts.nmr) state.charts.nmr.destroy();
    state.charts.nmr = new Chart(ctx, {
      type: 'line',
      plugins: [nmrSticks],
      data: {
        labels: xs,
        datasets: [{
          label: 'Intensity',
          data: ys,
          borderColor: 'rgba(16,120,110,1)',
          backgroundColor: 'rgba(16,120,110,.15)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            reverse: true,
            title: { display: true, text: 'Chemical shift (ppm)' },
            ticks: {
              maxTicksLimit: 12,
              // The labels are the raw ppm grid, so Chart.js printed every
              // digit of every one of them -- 7.135857657177138 and
              // 7.135857657177101 as two separate ticks.  Twelve of those in
              // a 300 px axis overlap into a band that cannot be read.
              // Only the *displayed* text is rounded: data.labels is left
              // alone because the stick plugin maps shifts through it, and
              // rounding there would collapse neighbouring grid points onto
              // one category.
              callback(value) {
                const raw = this.getLabelForValue(value);
                const n = Number(raw);
                return Number.isFinite(n) ? n.toFixed(2) : raw;
              },
            },
          },
          y: {
            title: { display: true, text: 'Relative intensity' },
            beginAtZero: true,
          },
        },
      },
    });
    state.charts.nmr.$nmrSticks = spec.sticks || [];
  }

  const note = $('#nmrNote');
  if (note) {
    const lw = Number(spec.linewidth_hz || 1);
    const lwReq = Number(spec.linewidth_hz_requested || lw);
    const mhz = Number(spec.larmor_mhz || spec.spectrometer_mhz || 400);
    const groups = (spec.sticks || []);
    if (!groups.length) {
      note.innerHTML = 'Only absolute shieldings are available for this '
        + 'molecule - no reference compound could be computed at this level, '
        + 'so there is no shift axis to draw.';
    } else {
      const parts = [];
      parts.push(`${escapeHtml(nmrIsotope || 'all')} spectrum: `
        + `${groups.length} signal${groups.length === 1 ? '' : 's'} at `
        + `${mhz.toFixed(1)} MHz`);
      // The drawn width is the one that matters, and when the cap forced a
      // wider line the figure must say so rather than name the requested one.
      parts.push(lwReq !== lw
        ? `Lorentzian ${lw.toFixed(3)} Hz (widened from the requested `
          + `${lwReq.toFixed(3)} Hz to fit the grid)`
        : `Lorentzian ${lw.toFixed(3)} Hz (${Number(spec.linewidth_ppm || 0)
          .toFixed(4)} ppm)`);
      parts.push(`grid ${Number(spec.grid_step_ppm || 0).toFixed(5)} ppm, `
        + `${Number(spec.points_per_hwhm || 0).toFixed(1)} points per `
        + 'half-width');
      parts.push('the red sticks are the signals, drawn at the integral of '
        + 'each group (height relative to the tallest); the curve is the same '
        + 'signals broadened');
      note.innerHTML = parts.join(' &middot; ') + '.';
    }
  }

  // ---- the shifts a paper quotes ----------------------------------------
  const groups = r.groups || [];
  $('#nmrTable').innerHTML =
    '<thead><tr><th>Nucleus</th><th>Shift (ppm)</th>'
    + '<th>Absolute shielding (ppm)</th><th>Count</th>'
    + '<th>Spread (ppm)</th></tr></thead><tbody>'
    + (groups.length
      ? groups.map((g) => `<tr><td>${escapeHtml(g.isotope || g.element)}</td>`
        + `<td>${Number(g.delta_ppm).toFixed(2)}</td>`
        + `<td>${Number(g.sigma_iso_ppm).toFixed(2)}</td>`
        + `<td>${g.count}</td>`
        + `<td>${Number(g.spread_ppm).toFixed(3)}</td></tr>`).join('')
      : '<tr><td colspan="5" class="muted">no reference available</td></tr>')
    + '</tbody>';

  // ---- tensors -----------------------------------------------------------
  const tn = $('#nmrTensorNote');
  if (tn) {
    tn.innerHTML = 'Principal values are SHIELDINGS (\u03C3), ascending. '
      + 'A shift anisotropy is the same three numbers with the opposite sign, '
      + 'because \u03B4 = \u03C3<sub>ref</sub> \u2212 \u03C3. '
      + 'Skew is 3(\u03C3<sub>22</sub> \u2212 \u03C3<sub>iso</sub>) / span.';
  }
  $('#nmrTensor').innerHTML =
    '<thead><tr><th>Atom</th><th>\u03C3<sub>11</sub></th>'
    + '<th>\u03C3<sub>22</sub></th><th>\u03C3<sub>33</sub></th>'
    + '<th>Span</th><th>Skew</th><th>\u0394\u03C3</th>'
    + '<th>Asymmetry</th></tr></thead><tbody>'
    + nuclei.map((n) => {
      const e = n.eigenvalues_ppm || [0, 0, 0];
      const asym = Number(n.tensor_asymmetry_ppm || 0);
      return `<tr><td>${escapeHtml(n.symbol)}${n.index + 1}`
        + (n.nmr_active ? '' : ' <span class="muted">(inactive)</span>') + '</td>'
        + `<td>${Number(e[0]).toFixed(2)}</td>`
        + `<td>${Number(e[1]).toFixed(2)}</td>`
        + `<td>${Number(e[2]).toFixed(2)}</td>`
        + `<td>${Number(n.span_ppm).toFixed(2)}</td>`
        + `<td>${Number(n.skew).toFixed(3)}</td>`
        + `<td>${Number(n.anisotropy_ppm).toFixed(2)}</td>`
        + `<td${asym > 0.01 ? ' class="warn"' : ''}>${asym.toFixed(3)}</td>`
        + '</tr>';
    }).join('')
    + '</tbody>';
}

/* ---- Raman ------------------------------------------------------------- */
// Drawn on its own axes rather than overlaid on the IR curve: the two spectra
// are both normalised to their own maximum, so a shared scale would make
// water's symmetric stretch look equally strong in both when it is the
// strongest Raman line and the weakest IR band by a factor of 44.
function renderRaman(r, scaled) {
  const ram = r.raman || {};
  const spec = r.raman_spectrum;
  const wrap = $('#ramanWrap');
  const skip = $('#ramanSkip');
  if (!wrap || !skip) return;

  if (ram.available === false) {
    wrap.classList.add('hidden');
    skip.classList.remove('hidden');
    skip.textContent = `Raman spectrum not computed: ${ram.reason || ''}`;
    return;
  }
  if (!spec || !spec.x || !spec.x.length) {
    wrap.classList.add('hidden');
    skip.classList.add('hidden');
    return;
  }
  skip.classList.add('hidden');
  wrap.classList.remove('hidden');

  const ctx = $('#ramanChart').getContext('2d');
  if (state.charts.raman) state.charts.raman.destroy();
  state.charts.raman = new Chart(ctx, {
    type: 'line',
    data: {
      labels: spec.x,
      datasets: [{
        label: 'Raman activity',
        data: spec.y,
        borderColor: 'rgba(147,51,234,1)',
        backgroundColor: 'rgba(147,51,234,.15)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: spec.xlabel || 'Wavenumber (cm-1)' } },
        y: {
          title: {
            display: true,
            text: spec.ylabel || 'Relative Raman intensity (%)',
          },
          beginAtZero: true,
        },
      },
    },
  });

  const note = $('#ramanNote');
  const bits = [];
  const iso = Number(ram.alpha_iso_au);
  if (Number.isFinite(iso)) {
    bits.push(`\u03B1<sub>iso</sub> = <b>${iso.toFixed(3)} a.u.</b> `
      + `(${Number(ram.alpha_iso_a3).toFixed(3)} \u00C5<sup>3</sup>)`);
  }
  const rs = ram.sum_rule || {};
  if (rs.residual_pct !== undefined) {
    const ok = Math.abs(Number(rs.residual_pct)) <= 0.5;
    bits.push(ok
      ? `absolute activities: sum rule closes to ${rs.residual_pct}%`
      : `<strong class="warn">sum rule residual ${rs.residual_pct}%</strong>`);
  }
  // The 3/4 ceiling is a theorem, so it is worth showing the reader where the
  // computed ratios sit against it rather than just listing them.
  const rho = ram.depolarization || [];
  if (rho.length) {
    const worst = Math.max.apply(null, rho.map(Number));
    bits.push(`max \u03C1 = ${worst.toFixed(3)} (the limit is 0.750)`);
  }
  if (ram.basis_warning) {
    bits.push(`<strong class="warn">basis caution</strong> `
      + escapeHtml(ram.basis_warning));
  }
  note.innerHTML = bits.join(' &middot; ');

  const act = ram.activities_a4_amu || [];
  const peaks = spec.peaks || [];
  $('#ramanTable').innerHTML =
    '<thead><tr><th>Harmonic (cm-1)</th>' +
    (scaled ? '<th>Scaled (cm-1)</th>' : '') +
    '<th>Raman activity (A<sup>4</sup>/amu)</th><th>\u03C1</th></tr></thead>'
    + '<tbody>' + peaks.map((q) => `<tr><td>${q.frequency_cm1}</td>` +
      (scaled ? `<td>${q.scaled_cm1}</td>` : '') +
      `<td>${Number(q.activity_a4_amu).toFixed(3)}</td>` +
      `<td>${Number(q.depolarization).toFixed(3)}</td></tr>`).join('')
    + '</tbody>';
  void act;
}

const DOS_COLORS = ['#1f6feb', '#d2691e', '#16a34a', '#9333ea', '#0891b2', '#b45309'];

function renderDos(r) {
  const d = r.dos;
  const box = $('#dosContent');
  if (!box || !d || !d.x_energy_ev) return;
  box.classList.remove('hidden');
  const p = document.querySelector('[data-pane="dos"] .empty-state');
  if (p) p.classList.add('hidden');

  const sets = [{
    label: 'Total',
    data: d.total,
    borderColor: '#111827',
    borderWidth: 1.8,
    pointRadius: 0,
    fill: false,
  }];
  let i = 0;
  Object.keys(d.projected || {}).forEach((el) => {
    sets.push({
      label: el,
      data: d.projected[el],
      borderColor: DOS_COLORS[i % DOS_COLORS.length],
      borderWidth: 1.2,
      pointRadius: 0,
      fill: true,
      backgroundColor: DOS_COLORS[i % DOS_COLORS.length] + '22',
    });
    i += 1;
  });

  const ctx = $('#dosChart').getContext('2d');
  if (state.charts.dos) state.charts.dos.destroy();
  state.charts.dos = new Chart(ctx, {
    type: 'line',
    data: { labels: d.x_relative_ev, datasets: sets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'bottom' } },
      scales: {
        x: { title: { display: true, text: 'Energy relative to HOMO (eV)' } },
        y: { title: { display: true, text: d.ylabel || 'Density of states' }, beginAtZero: true },
      },
    },
  });

  const dn = $('#dosNote');
  if (dn) {
    dn.textContent = d.note
      || `Gaussian broadening, FWHM ${d.fwhm_ev} eV; HOMO at ${d.homo_ev} eV.`;
  }

  $('#dosTable').innerHTML =
    '<thead><tr><th>HOMO (eV)</th><th>LUMO (eV)</th><th>Occupied orbitals</th>' +
    '<th>Broadening (eV)</th></tr></thead><tbody>' +
    `<tr><td>${d.homo_ev}</td><td>${d.lumo_ev}</td>` +
    `<td>${d.n_occupied}</td><td>${d.fwhm_ev}</td></tr></tbody>`;
}

function renderReactivity(r) {
  if (r.electrophilicity_ev === undefined) return;
  const f = r.fukui;
  const el = document.querySelector('[data-pane="narrative"]');
  if (!el || !f) return;
  const div = $('#reactivityBlock');
  if (!div) return;
  div.classList.remove('hidden');
  const g = [
    ['HOMO', r.homo_ev + ' eV'], ['LUMO', r.lumo_ev + ' eV'],
    ['Gap', r.gap_ev + ' eV'], ['Ionisation potential', r.ionization_potential_ev + ' eV'],
    ['Electron affinity', r.electron_affinity_ev + ' eV'],
    ['Chemical potential', r.chemical_potential_ev + ' eV'],
    ['Hardness', r.hardness_ev + ' eV'],
    ['Electrophilicity', r.electrophilicity_ev + ' eV'],
  ];
  let html = '<h3>Conceptual DFT reactivity</h3>' +
    g.map(([k, v]) => `<span class="chip"><b>${k}</b> ${v}</span>`).join('');
  html += '<div class="table-wrap"><table><thead><tr><th>Atom</th>' +
    '<th>f<sup>+</sup> (nucleophilic attack)</th><th>f<sup>-</sup> ' +
    '(electrophilic attack)</th><th>f<sup>0</sup></th></tr></thead><tbody>' +
    f.rows.map((q) => `<tr><td>${q.symbol}${q.atom}</td><td>${q.f_plus}</td>` +
      `<td>${q.f_minus}</td><td>${q.f_zero}</td></tr>`).join('') +
    '</tbody></table></div>';
  html += `<p class="muted">Preferred site for electrophilic attack: ` +
    `<b>${f.most_electrophilic_site}</b>; for nucleophilic attack: ` +
    `<b>${f.most_nucleophilic_site}</b>.</p>`;
  div.innerHTML = html;
}

/* ---- isosurfaces (MEP map, orbital and density contours) ---------------- */
const SURF_PHASE_POS = '#1f6feb';   // blue
const SURF_PHASE_NEG = '#d2691e';   // orange
const SURF_DENSITY = '#16a34a';

function surfaceLabel(it) {
  // An NTO entry has no canonical-orbital index: it is a rotated combination
  // of them, and printing "orbital NaN" was what happened when it was
  // assumed every orbital surface had one.
  if (it.kind === 'orbital') {
    if (it.index === undefined || it.index === null) return it.label;
    const e = it.energy_ev === undefined || it.energy_ev === null
      ? '' : ` (${it.energy_ev} eV)`;
    return `${it.label} — orbital ${it.index + 1}${e}`;
  }
  return it.label;
}

function renderNto(r) {
  const box = $('#ntoBlock');
  if (!box) return;
  const n = r.nto;
  if (!n) { box.classList.add('hidden'); box.innerHTML = ''; return; }
  box.classList.remove('hidden');

  const head = `State ${n.state}` +
    (n.energy_ev !== undefined && n.energy_ev !== null
      ? ` — ${n.energy_ev.toFixed(3)} eV` : '') +
    (n.wavelength_nm ? ` (${n.wavelength_nm.toFixed(1)} nm)` : '');
  const canon = (n.canonical || []).slice(0, 3).map((c) =>
    `<span class="nto-canon">${escapeHtml(c.from_label)}→` +
    `${escapeHtml(c.to_label)} <em>${(c.weight * 100).toFixed(1)}%</em></span>`)
    .join('');

  let html = `<div class="nto-head"><h3>Natural transition orbitals</h3>` +
    `<span class="nto-state">${escapeHtml(head)}</span></div>`;
  html += `<p class="nto-lead">The leading NTO pair carries ` +
    `<b>${((n.dominant_occupation || 0) * 100).toFixed(1)}%</b> of the ` +
    `transition — ${escapeHtml(n.character || '')}.</p>`;
  html += `<table class="nto-table"><thead><tr><th>NTO pair</th>` +
    `<th>Occupation</th><th>Hole (donor)</th><th>Particle (acceptor)</th>` +
    `</tr></thead><tbody>`;
  (n.pairs || []).forEach((p) => {
    html += `<tr><td>${p.pair}</td><td>${(p.occupation * 100).toFixed(1)}%</td>` +
      `<td>${escapeHtml(p.donor_file)}</td>` +
      `<td>${escapeHtml(p.acceptor_file)}</td></tr>`;
  });
  html += '</tbody></table>';
  if (canon) {
    html += `<div class="nto-canon-list"><span class="muted">In canonical ` +
      `orbitals:</span>${canon}</div>`;
  }
  box.innerHTML = html;
}

function renderSurfaces(r) {
  const s = r.surfaces;
  const box = $('#surfContent');
  if (!box || !s || !s.surfaces || !s.surfaces.length) return;

  state.surfaces = s;
  state.cubeCache = new Map();
  box.classList.remove('hidden');
  const p = document.querySelector('[data-pane="surfaces"] .empty-state');
  if (p) p.classList.add('hidden');

  const sel = $('#surfSelect');
  sel.innerHTML = s.surfaces
    .map((it) => `<option value="${it.key}">${escapeHtml(surfaceLabel(it))}</option>`)
    .join('');
  const def = s.surfaces.find((it) => it.default) || s.surfaces[0];
  sel.value = def.key;

  if (!state.surfWired) {
    state.surfWired = true;
    sel.addEventListener('change', () => {
      syncSurfaceControls();
      applySurface();
    });
    $('#surfIso').addEventListener('input', () => {
      $('#surfIsoVal').textContent = (+$('#surfIso').value).toFixed(3);
      applySurface();
    });
    $('#surfOpacity').addEventListener('input', () => {
      $('#surfOpacityVal').textContent = (+$('#surfOpacity').value).toFixed(2);
      applySurface();
    });
    $('#surfBothPhases').addEventListener('change', () => applySurface());
  }
  renderNto(r);
  syncSurfaceControls();
  applySurface();
}

function syncSurfaceControls() {
  const s = state.surfaces;
  const key = $('#surfSelect').value;
  const it = (s.surfaces || []).find((x) => x.key === key);
  if (!it) return;

  const isMap = it.kind === 'mep';
  // a map is coloured by value, so an isovalue is meaningless for it
  $('#surfIso').closest('.surf-field').classList.toggle('hidden', isMap);
  $('#surfBothPhases').closest('.surf-check')
    .classList.toggle('hidden', !it.bipolar);
  $('#surfIso').value = it.isoval || 0.032;
  $('#surfIsoVal').textContent = (+$('#surfIso').value).toFixed(3);

  const legend = $('#surfLegend');
  if (isMap && it.range) {
    legend.classList.remove('hidden');
    $('#surfLegendLo').textContent =
      `${it.range[0]} Ha/e (${it.range_kcal_mol[0]} kcal/mol)`;
    $('#surfLegendHi').textContent =
      `+${it.range[1]} Ha/e (+${it.range_kcal_mol[1]} kcal/mol)`;
  } else {
    legend.classList.add('hidden');
  }
  $('#surfNote').textContent = it.note || '';
}

function applySurface() {
  const s = state.surfaces;
  if (!s || !state.viewer) return;
  const key = $('#surfSelect').value;
  const it = (s.surfaces || []).find((x) => x.key === key);
  if (!it) return;

  const status = $('#surfStatus');
  const url = `${s.url_base}/${it.file}`;
  status.className = 'surf-status busy';
  status.textContent = 'loading grid…';

  const cached = state.cubeCache.get(url);
  const work = cached
    ? Promise.resolve(cached)
    : fetch(url).then((resp) => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.text();
      }).then((text) => {
        state.cubeCache.set(url, text);
        return text;
      });

  work.then((text) => drawSurface(it, text, status))
    .catch((err) => {
      status.className = 'surf-status error';
      status.textContent = `could not load ${it.file}: ${err.message}`;
    });
}

function drawSurface(it, text, status) {
  const viewer = state.viewer;
  // every redraw starts clean: 3Dmol stacks shapes, so re-running without
  // clearing piles up translucent shells
  viewer.removeAllShapes();
  viewer.removeAllSurfaces();

  const opacity = +$('#surfOpacity').value || 0.85;

  if (it.kind === 'mep') {
    // Colour the van der Waals surface by the potential underneath it.
    // RWB runs red -> white -> blue, and the range is negative-first, which
    // is the convention papers use: red is electron-rich.
    const lo = it.range ? it.range[0] : -0.05;
    const hi = it.range ? it.range[1] : 0.05;
    viewer.addSurface($3Dmol.SurfaceType.VDW, {
      voldata: text,
      volformat: 'cube',
      volscheme: new $3Dmol.Gradient.RWB(lo, hi),
      opacity: opacity,
    }, {});
  } else {
    const isoval = +$('#surfIso').value || 0.032;
    const both = it.bipolar && $('#surfBothPhases').checked;
    const color = it.kind === 'density' ? SURF_DENSITY : SURF_PHASE_POS;
    const spec = {
      isoval: isoval,
      color: color,
      opacity: opacity,
      smoothness: 2,
    };
    viewer.addVolumetricData(text, 'cube', spec);
    if (both) {
      viewer.addVolumetricData(text, 'cube', Object.assign({}, spec, {
        isoval: -isoval,
        color: SURF_PHASE_NEG,
      }));
    }
  }

  viewer.render();
  status.className = 'surf-status';
  const grid = state.surfaces.grid_n;
  status.textContent =
    `${it.label} · ${grid}³ grid · ` +
    (it.kind === 'mep'
      ? `range ±${Math.abs(it.range[1])} Ha/e`
      : `isovalue ${(+$('#surfIso').value).toFixed(3)} ${it.units}` +
        (it.bipolar && $('#surfBothPhases').checked ? ' (both phases)' : ''));
}

/* ---- figure export ------------------------------------------------------ */
/* A screenshot of a web canvas is not a paper figure: it is 96 dpi, it has
   no background, and the numbers behind it are lost.  So every chart gets a
   PNG (re-rendered at 3x, on white) and a CSV (long format, one row per
   point) so the curve can be redrawn at publication resolution.            */

function chartById(id) {
  for (const key of Object.keys(state.charts)) {
    const c = state.charts[key];
    if (c && c.canvas && c.canvas.id === id) return c;
  }
  return null;
}

function exportChartPng(chart, name) {
  if (!chart) return;
  const cv = chart.canvas;
  const w = Math.max(320, cv.clientWidth || cv.width);
  const h = Math.max(200, cv.clientHeight || cv.height);
  const scale = 3;
  // Re-render at the larger size rather than upscaling the bitmap, so lines
  // and text stay crisp instead of being interpolated.
  const oldW = cv.style.width, oldH = cv.style.height;
  let out;
  try {
    cv.style.width = (w * scale) + 'px';
    cv.style.height = (h * scale) + 'px';
    chart.resize(w * scale, h * scale);
    const off = document.createElement('canvas');
    off.width = w * scale; off.height = h * scale;
    const c = off.getContext('2d');
    c.fillStyle = '#ffffff';
    c.fillRect(0, 0, off.width, off.height);
    c.drawImage(cv, 0, 0);
    out = off.toDataURL('image/png');
  } finally {
    cv.style.width = oldW; cv.style.height = oldH;
    chart.resize();
  }
  saveDataUrl(out, name + '.png');
}

function exportChartCsv(chart, name) {
  if (!chart) return;
  const d = chart.data;
  const rows = [['series', 'x', 'y']];
  (d.datasets || []).forEach((ds) => {
    const label = String(ds.label || 'series').replace(/"/g, '""');
    (ds.data || []).forEach((pt, i) => {
      // scatter data is {x, y}; line/bar data is a number keyed by the label
      let x, y;
      if (pt !== null && typeof pt === 'object') { x = pt.x; y = pt.y; }
      else { x = (d.labels && d.labels[i] !== undefined) ? d.labels[i] : i; y = pt; }
      if (x === null || x === undefined || y === null || y === undefined) return;
      rows.push([`"${label}"`, x, y]);
    });
  });
  saveDataUrl('data:text/csv;charset=utf-8,' +
              encodeURIComponent(rows.map((r) => r.join(',')).join('\n')),
              name + '.csv');
}

function saveDataUrl(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/* ---- NCI / reduced density gradient ------------------------------------ */
function renderNCI(r) {
  const d = r.nci;
  const box = $('#nciContent');
  if (!box || !d || !d.scatter) return;

  state.nci = d;
  if (!state.cubeCache) state.cubeCache = new Map();
  box.classList.remove('hidden');
  const p = document.querySelector('[data-pane="nci"] .empty-state');
  if (p) p.classList.add('hidden');

  // ---- the scatter: the figure itself --------------------------------
  const sc = d.scatter;
  if (state.charts.nci) state.charts.nci.destroy();
  const ctx = document.getElementById('nciChart');
  state.charts.nci = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: (sc.series || []).filter((s) => (s.points || []).length).map((s) => ({
        label: `${s.label} (${s.count.toLocaleString()} pts)`,
        data: s.points.map((pt) => ({ x: pt[0], y: pt[1] })),
        backgroundColor: s.colour,
        pointRadius: 1.6,
        pointHoverRadius: 4,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { labels: { boxWidth: 10, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (c) => `sign(l2)rho ${c.parsed.x.toFixed(4)}, ` +
                          `RDG ${c.parsed.y.toFixed(3)}`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: sc.x_label },
          min: sc.x_range ? sc.x_range[0] : -0.05,
          max: sc.x_range ? sc.x_range[1] : 0.05,
        },
        // RDG starts at 0 and grows, so the y axis runs downwards in the
        // sense that an interaction is a tongue reaching *down* to zero.
        y: {
          title: { display: true, text: sc.y_label },
          min: sc.y_range ? sc.y_range[0] : 0,
          max: sc.y_range ? sc.y_range[1] : 2,
          reverse: false,
        },
      },
    },
  });

  // ---- one line of numbers under the plot ----------------------------
  const sum = $('#nciSummary');
  const bits = [];
  const g = d.grid || {};
  if (g.points) {
    bits.push(`<span>${g.n}&sup3; grid &middot; ` +
              `${g.points.toLocaleString()} points</span>`);
  }
  if (d.min_rdg !== null && d.min_rdg !== undefined) {
    bits.push(`<span>lowest RDG <strong>${d.min_rdg.toFixed(3)}</strong></span>`);
  }
  if (d.sign_l2_rho_at_spike !== null && d.sign_l2_rho_at_spike !== undefined) {
    bits.push(`<span>at sign(&lambda;&#8322;)&rho; ` +
              `<strong>${d.sign_l2_rho_at_spike.toFixed(4)}</strong></span>`);
  }
  if (d.n_spike_points !== undefined) {
    bits.push(`<span>${d.n_spike_points} spike points</span>`);
  }
  bits.push(`<span class="${d.has_weak_interaction ? 'ok' : 'muted'}">` +
            (d.has_weak_interaction
              ? 'weak interaction detected'
              : 'no weak-interaction spike') +
            '</span>');
  sum.innerHTML = bits.join('');

  // ---- the 3D isosurface ---------------------------------------------
  const sel = $('#nciSelect');
  const opts = (d.surfaces || []).filter((s) => s.kind === 'nci');
  sel.innerHTML = opts
    .map((it) => `<option value="${it.key}">${escapeHtml(it.label)}</option>`)
    .join('');
  const def = opts.find((it) => d.has_weak_interaction &&
                                it.key === 'nci_attractive') || opts[0];
  if (def) sel.value = def.key;

  if (!state.nciWired && opts.length) {
    state.nciWired = true;
    sel.addEventListener('change', () => applyNCISurface());
    $('#nciIso').addEventListener('input', () => {
      $('#nciIsoVal').textContent = (+$('#nciIso').value).toFixed(2);
      applyNCISurface();
    });
    $('#nciOpacity').addEventListener('input', () => {
      $('#nciOpacityVal').textContent = (+$('#nciOpacity').value).toFixed(2);
      applyNCISurface();
    });
  }
  if (opts.length) {
    $('#nciIso').value = (opts[0].isoval || 0.5).toFixed(2);
    $('#nciIsoVal').textContent = (+$('#nciIso').value).toFixed(2);
    applyNCISurface();
  }
  $('#nciNote').textContent = opts.length
    ? opts[0].note || ''
    : 'No isosurface grids were written for this run.';
}

/* ---- series: one level of theory, many molecules --------------------- */
const SERIES_PALETTE = [
  '#1f6feb', '#d2691e', '#16a34a', '#9333ea', '#b45309', '#0891b2',
  '#be185d', '#4b5563', '#65a30d', '#ca8a04', '#0d9488', '#7c3aed',
];

function renderSeries(job) {
  const s = job.result;
  if (!s || !s.points || !s.points.length) return;
  state.series = s;

  $('#emptySeries')?.classList.add('hidden');
  $('#seriesContent')?.classList.remove('hidden');

  const level = `${s.functional_label || s.functional || ''} / ${s.basis_label || s.basis || ''}`
    .trim(' /');
  $('#seriesTitle').textContent = `${s.points.length} molecules at ${level}`;

  const bits = [`${s.n_ok} calculated`];
  if (s.n_failed) bits.push(`${s.n_failed} failed`);
  if (s.solvation) bits.push(`solvation: ${escapeHtml(String(s.solvation))}`);
  const secs = (s.points || [])
    .map((p) => p.scf_seconds).filter((v) => typeof v === 'number');
  if (secs.length) {
    bits.push(`${secs.reduce((a, b) => a + b, 0).toFixed(1)} s of SCF`);
  }
  // A//B: the level of theory is only half of what makes these comparable.
  if (s.geometry_source) {
    bits.push(`geometry: ${escapeHtml(String(s.geometry_source))}`);
  }
  $('#seriesSub').innerHTML = bits.join(' &middot; ');

  // property picker: only offer columns that actually carry numbers, so the
  // selector can never produce an empty chart
  const props = (s.properties || []).filter((p) =>
    (s.points || []).some((pt) => pt[p.key] !== null && pt[p.key] !== undefined));
  const sel = $('#seriesProp');
  const prev = state.seriesProp;
  sel.innerHTML = props.map((p) =>
    `<option value="${escapeHtml(p.key)}">${escapeHtml(p.label)}` +
    (p.unit ? ` (${escapeHtml(p.unit)})` : '') + '</option>').join('');
  if (prev && props.some((p) => p.key === prev)) sel.value = prev;
  state.seriesProps = props;

  renderSeriesTable(s, props);
  drawSeriesChart();

  const notes = [];
  if (s.n_failed) {
    notes.push(`${s.n_failed} of ${s.points.length} molecules could not be ` +
      'calculated; they are kept in the table with the reason, because ' +
      'dropping them would make the trend look like a real gap in the data.');
  }
  const geom = String(s.geometry_source || '');
  notes.push('Every point is a real single point at the level of theory ' +
    'above, so the numbers are comparable across the set; total energies ' +
    'are not comparable between different formulas, which is why the ' +
    'relative column is given only as a convenience.');
  if ((s.n_ok || 0) > 12) {
    notes.push('The points are joined only in the order they were requested; ' +
      'there is no continuous variable between one molecule and the next, so ' +
      'the line is a reading aid and not a trend.');
  }
  if (geom && !/dft|optimis|optimiz/i.test(geom)) {
    notes.push(`These are single points on ${geom} geometries, not on ` +
      'minima at this level of theory \u2014 quote them as ' +
      `${level}//${geom.split('(')[0].trim()}. Gap and dipole both move when ` +
      'the geometry is relaxed, so optimise before putting a number from ' +
      'this set in a table.');
  }
  $('#seriesNote').textContent = notes.join(' ');

  if (job.explanation) {
    addMessage('assistant', markdown(job.explanation));
  }
  switchTab('series');
}

function seriesPropDef() {
  const props = state.seriesProps || [];
  const key = state.seriesProp;
  return props.find((p) => p.key === key) || props[0] || null;
}

function drawSeriesChart() {
  const s = state.series;
  if (!s) return;
  const def = seriesPropDef();
  if (!def) return;

  const ok = s.points.filter((p) =>
    p[def.key] !== null && p[def.key] !== undefined);
  const labels = ok.map((p) => p.name || p.spec);
  const values = ok.map((p) => Number(p[def.key]));
  const colors = ok.map((_, i) => SERIES_PALETTE[i % SERIES_PALETTE.length]);

  const ctx = $('#seriesChart').getContext('2d');
  if (state.charts.series) state.charts.series.destroy();

  // more than a dozen bars makes the labels unreadable whatever we do, so
  // switch to points joined by a line and keep the value in the tooltip
  const many = ok.length > 12;
  state.charts.series = new Chart(ctx, {
    type: many ? 'line' : 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: def.unit ? `${def.label} (${def.unit})` : def.label,
        data: values,
        backgroundColor: many ? 'rgba(31,111,235,.12)' : colors,
        borderColor: many ? '#1f6feb' : colors.map((c) => c),
        borderWidth: many ? 2 : 1,
        pointBackgroundColor: colors,
        pointRadius: many ? 4 : 0,
        // No fill, even in the line form: an area under a curve implies an
        // integral over the x axis, and this x axis is a list of molecules.
        fill: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: `${def.label}` + (def.unit ? ` (${def.unit})` : '') +
            ` — ${s.functional_label || s.functional} / ${s.basis_label || s.basis}`,
          color: '#46556b',
          font: { size: 11, weight: 'normal' },
        },
        tooltip: {
          callbacks: {
            label: (c) => `${c.parsed.y.toFixed(def.decimals)} ` +
              (def.unit || ''),
          },
        },
      },
      scales: {
        y: {
          title: {
            display: true,
            text: def.unit ? `${def.label} (${def.unit})` : def.label,
            color: '#7a8797', font: { size: 11 },
          },
          ticks: { color: '#7a8797', font: { size: 10 } },
          grid: { color: '#e4e9f0' },
        },
        x: {
          ticks: {
            color: '#7a8797', font: { size: 10 },
            maxRotation: 90, minRotation: labels.length > 6 ? 45 : 0,
          },
          grid: { display: false },
        },
      },
    },
  });
}

function renderSeriesTable(s, props) {
  const table = $('#seriesTable');
  let html = '<thead><tr><th>#</th><th>Molecule</th><th>Formula</th>' +
    '<th>Atoms</th>';
  props.forEach((p) => {
    html += `<th>${escapeHtml(p.label)}` +
      (p.unit ? ` <span class="unit">${escapeHtml(p.unit)}</span>` : '') +
      '</th>';
  });
  html += '<th>Status</th></tr></thead><tbody>';

  s.points.forEach((p, i) => {
    const bad = p.gap_ev === null || p.gap_ev === undefined;
    html += `<tr${bad ? ' class="row-failed"' : ''}>`
      + `<td>${i + 1}</td>`
      + `<td>${escapeHtml(p.name || p.spec)}</td>`
      + `<td>${escapeHtml(p.formula || '—')}</td>`
      + `<td>${p.natoms === null || p.natoms === undefined ? '—' : p.natoms}</td>`;
    props.forEach((d) => {
      const v = p[d.key];
      html += '<td>' + (v === null || v === undefined
        ? '—'
        : Number(v).toFixed(d.decimals)) + '</td>';
    });
    html += `<td>${bad
      ? `<span class="tag-failed">failed</span> ` +
        `<span class="err">${escapeHtml(p.error || 'unknown')}</span>`
      : (p.converged
        ? '<span class="tag-ok">converged</span>'
        : '<span class="tag-warn">not converged</span>')}</td>`;
    html += '</tr>';
  });
  table.innerHTML = html + '</tbody>';
}

function seriesToCsv() {
  const s = state.series;
  if (!s) return;
  const props = state.seriesProps || [];
  const head = ['molecule', 'spec', 'smiles', 'formula', 'atoms', 'electrons',
    ...props.map((p) => p.key + (p.unit ? '_' + p.unit : '')),
    'converged', 'error'];
  const q = (v) => {
    const t = v === null || v === undefined ? '' : String(v);
    return /[",\n]/.test(t) ? '"' + t.replace(/"/g, '""') + '"' : t;
  };
  const rows = [head.join(',')];
  s.points.forEach((p) => {
    rows.push([
      p.name, p.spec, p.smiles, p.formula, p.natoms, p.nelectrons,
      ...props.map((d) => (p[d.key] === null || p[d.key] === undefined
        ? '' : Number(p[d.key]).toFixed(d.decimals))),
      p.converged ? 'yes' : 'no', p.error || '',
    ].map(q).join(','));
  });
  // a level-of-theory line at the top, so the file still says what it is
  // six months after it was downloaded
  const meta = `# ${s.functional_label || s.functional} / ` +
    `${s.basis_label || s.basis}` +
    (s.solvation ? `, solvation=${s.solvation}` : '');
  saveDataUrl('data:text/csv;charset=utf-8,' +
    encodeURIComponent(meta + '\n' + rows.join('\n')), 'series.csv');
}

/* ---- reaction paths: the energy profile every mechanism paper opens with */
function renderReaction(job) {
  const r = job.result;
  if (!r || !r.points || !r.points.length) return;
  state.reaction = r;

  $('#emptyReaction')?.classList.add('hidden');
  $('#reactionContent')?.classList.remove('hidden');

  const coord = r.coordinate || {};
  const unit = coord.unit || '';
  const level = `${r.functional_label || r.functional || ''} / ` +
    `${r.basis_label || r.basis || ''}`.trim(' /');

  $('#reactionTitle').textContent =
    `Reaction path along ${coord.label || 'the coordinate'}`;

  const bits = [`${r.n_points} relaxed points`, level];
  if (typeof r.seconds === 'number') bits.push(`${r.seconds.toFixed(0)} s`);
  if (r.solvation) bits.push(`solvation: ${String(r.solvation)}`);
  $('#reactionSub').innerHTML = bits.map(escapeHtml).join(' &middot; ');

  // ---- verdict badge: the whole point of the Hessian check -----------
  const ver = r.verification || {};
  const badge = $('#reactionVerdict');
  if (ver.checked && ver.is_transition_state) {
    badge.className = 'tag tag-ok';
    badge.textContent = 'TS verified \u00b7 1 imaginary mode';
  } else if (ver.checked) {
    badge.className = 'tag tag-warn';
    badge.textContent = `not a verified TS \u00b7 ${ver.n_imaginary} imaginary`;
  } else if (r.barrier_bracketed) {
    badge.className = 'tag tag-warn';
    badge.textContent = 'TS not checked';
  } else {
    badge.className = 'tag tag-failed';
    badge.textContent = 'no barrier in window';
  }
  badge.classList.remove('hidden');

  // ---- headline numbers ---------------------------------------------
  const ev = (k) => (k === null || k === undefined ? null : k / 23.0605);
  const cells = [];
  if (r.barrier_kcal !== null && r.barrier_kcal !== undefined) {
    cells.push(['Barrier', `${r.barrier_kcal.toFixed(2)} kcal/mol`,
      `${ev(r.barrier_kcal).toFixed(3)} eV`]);
  } else {
    cells.push(['Barrier', '\u2014', 'not bracketed']);
  }
  cells.push(['Reaction energy',
    `${Number(r.reaction_kcal).toFixed(2)} kcal/mol`,
    `${ev(r.reaction_kcal).toFixed(3)} eV`]);
  if (r.ts && r.ts.coord !== null && r.ts.coord !== undefined) {
    cells.push(['Transition state',
      `${Number(r.ts.coord).toFixed(2)} ${unit}`,
      `${Number(r.ts.relative_kcal).toFixed(2)} kcal/mol`]);
  }
  if (ver.checked && ver.imaginary_cm && ver.imaginary_cm.length) {
    cells.push(['Imaginary mode', `${ver.imaginary_cm[0].toFixed(1)} cm\u207b\u00b9`,
      `overlap ${Number(ver.mode_alignment).toFixed(2)}`]);
  }
  $('#reactionMetrics').innerHTML = cells.map(([label, value, sub]) =>
    `<div class="metric"><span class="metric-label">${escapeHtml(label)}</span>` +
    `<span class="metric-value">${escapeHtml(value)}</span>` +
    `<span class="metric-sub">${escapeHtml(sub)}</span></div>`).join('');

  renderReactionTable(r);
  drawReactionChart();
  renderIrc(r);

  // ---- the barrier, in the several quantities papers actually quote ----
  // A relaxed scan gives an electronic energy difference with frozen nuclei.
  // Quoting that as an activation energy is wrong by the zero-point term,
  // which is 1-3 kcal/mol -- the size of the whole barrier for a torsion --
  // so the corrected values are shown beside it whenever they were computed,
  // and their absence is stated when they were not.
  const barBox = $('#reactionBarrier');
  if (barBox) {
    const bt = r.barrier_thermo || {};
    const rows = [['Electronic', r.barrier_kcal]];
    if (bt.computed) {
      rows.push(['Zero-point corrected', bt.zpe_corrected_kcal]);
      rows.push([`\u0394H\u2021 (${bt.temperature_K} K)`, bt.enthalpy_kcal]);
      rows.push([`\u0394G\u2021 (${bt.temperature_K} K)`, bt.gibbs_kcal]);
    }
    const kv = (k, v) => `<tr><th>${escapeHtml(k)}</th><td>${
      (v === null || v === undefined || Number.isNaN(Number(v)))
        ? '\u2014' : `${Number(v).toFixed(2)} kcal/mol`}</td></tr>`;
    let html = '<table class="kv">' + rows.map(([k, v]) => kv(k, v)).join('') +
      '</table>';
    if (bt.computed) {
      html += `<p class="muted">Harmonic oscillator / rigid rotor at ` +
        `${escapeHtml(String(bt.temperature_K))} K and 1 atm. The ` +
        `transition state's imaginary mode is left out of its partition ` +
        `function, so it has ${bt.transition_state.n_real_modes} real modes ` +
        `against the reactant's ${bt.reactant.n_real_modes}.</p>`;
    } else if (bt.reason) {
      html += `<p class="muted">Electronic only \u2014 ` +
        `${escapeHtml(bt.reason)}.</p>`;
    }
    barBox.innerHTML = html;
  }

  // ---- the TS geometry and the frequency list ------------------------
  const tsBox = $('#reactionTs');
  if (r.ts && r.ts.xyz) {
    tsBox.innerHTML =
      `<pre class="xyz-block">${escapeHtml(r.ts.xyz.trim())}</pre>` +
      `<button class="btn tiny ghost" id="reactionShowTs" type="button">` +
      `Show in viewer</button>`;
    $('#reactionShowTs')?.addEventListener('click', () => {
      showMolecule(r.molecule, { xyz: r.ts.xyz });
      switchTab('geometry');
    });
  } else {
    tsBox.innerHTML = '<p class="muted">No maximum was found inside the ' +
      'scanned window, so there is no transition-state geometry to show.</p>';
  }

  const checkBox = $('#reactionCheck');
  if (ver.checked) {
    const rows = [
      ['Imaginary frequencies', `${ver.n_imaginary}`],
      ['Lowest', (ver.imaginary_cm && ver.imaginary_cm.length)
        ? `${ver.imaginary_cm[0].toFixed(1)} cm\u207b\u00b9` : '\u2014'],
      ['Mode \u2194 coordinate overlap',
        `${Number(ver.mode_alignment).toFixed(3)}`],
      ['Verdict', ver.is_transition_state
        ? 'first-order saddle' : 'not a transition state'],
    ];
    checkBox.innerHTML = '<table class="kv">' + rows.map(([k, v]) =>
      `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(v)}</td></tr>`).join('') +
      '</table>';
  } else if (ver.error) {
    checkBox.innerHTML = `<p class="muted">${escapeHtml(ver.error)}</p>`;
  } else {
    checkBox.innerHTML = `<p class="muted">${escapeHtml(
      ver.reason || 'The frequency check did not run.')}</p>`;
  }

  const notes = [];
  if (ver.checked && !ver.is_transition_state) {
    notes.push('The maximum of the scan is not a verified transition state, ' +
      'so the barrier above is the height of the relaxed path \u2014 an upper ' +
      'bound on the true activation energy, not the saddle value.');
  }
  if (ver.checked && ver.is_transition_state) {
    notes.push('The maximum carries exactly one imaginary frequency whose ' +
      'eigenvector points along the reaction coordinate, which is what makes ' +
      'it a transition state. The barrier drawn on the profile is the ' +
      'electronic one; the zero-point and Gibbs values beside it are the ' +
      'ones to quote as an activation energy.');
  }
  if (!r.barrier_bracketed) {
    notes.push('No maximum was found inside the scanned window. A monotonic ' +
      'profile is a real result: it says this coordinate is not the one the ' +
      'reaction follows, or the window is too narrow.');
  }
  notes.push('Every point is a converged SCF at a geometry optimised with ' +
    'that coordinate held fixed, and the two endpoints are fully relaxed, so ' +
    'the reaction energy is between two genuine minima.');
  $('#reactionNote').textContent = notes.join(' ');

  if (job.explanation) {
    addMessage('assistant', markdown(job.explanation));
  }
  switchTab('reaction');
}

function drawReactionChart() {
  const r = state.reaction;
  if (!r) return;
  const coord = r.coordinate || {};
  const unit = coord.unit || '';

  // The scan points, plus the refined transition state, sorted along the
  // coordinate.  The TS usually falls *between* two scan points, so without
  // adding it the drawn peak would sit below the barrier that is reported.
  const series = (r.points || []).map((p) => ({
    x: Number(p.coord), y: Number(p.relative_kcal), ts: false,
  }));
  if (r.ts && r.ts.coord !== null && r.ts.coord !== undefined) {
    series.push({ x: Number(r.ts.coord), y: Number(r.ts.relative_kcal), ts: true });
  }
  series.sort((a, b) => a.x - b.x);

  const tsPoint = series.find((p) => p.ts) || null;
  const barrier = r.barrier_kcal;

  const ctx = $('#reactionChart').getContext('2d');
  if (state.charts.reaction) state.charts.reaction.destroy();

  // Dashed guides to the barrier and to the reactant well, drawn by hand:
  // the annotation plugin is not loaded, and an energy profile without them
  // makes the reader measure the barrier off the axis.
  const guides = {
    id: 'reactionGuides',
    afterDatasetsDraw(chart) {
      const { ctx: c, chartArea, scales } = chart;
      if (!chartArea || !scales.y) return;
      const yOf = (v) => scales.y.getPixelForValue(v);
      const xOf = (v) => scales.x.getPixelForValue(v);
      c.save();
      c.setLineDash([5, 4]);
      c.lineWidth = 1;

      if (tsPoint && barrier !== null && barrier !== undefined) {
        // the barrier height, measured from the reactant well (y = 0 here)
        c.strokeStyle = 'rgba(214,96,77,.85)';
        const yb = yOf(tsPoint.y);
        c.beginPath();
        c.moveTo(chartArea.left, yb);
        c.lineTo(chartArea.right, yb);
        c.stroke();
        c.setLineDash([]);
        c.fillStyle = 'rgba(214,96,77,1)';
        c.font = '600 12px system-ui, sans-serif';
        c.textAlign = 'left';
        c.textBaseline = 'bottom';
        c.fillText(`barrier ${Number(barrier).toFixed(2)} kcal/mol`,
          chartArea.left + 6, yb - 4);
      }

      if (tsPoint) {
        c.setLineDash([5, 4]);
        c.strokeStyle = 'rgba(120,120,120,.6)';
        const xt = xOf(tsPoint.x);
        c.beginPath();
        c.moveTo(xt, chartArea.top);
        c.lineTo(xt, chartArea.bottom);
        c.stroke();
      }
      c.restore();
    },
  };

  state.charts.reaction = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: 'Energy',
        data: series,
        borderColor: '#1f6feb',
        borderWidth: 2,
        // straight segments: the curve is a set of computed points, and a
        // smoothed spline would invent values that were never calculated
        tension: 0,
        fill: false,
        pointRadius: series.map((p) => (p.ts ? 6 : 3.5)),
        pointBackgroundColor: series.map((p) =>
          (p.ts ? '#d6604d' : '#1f6feb')),
        pointBorderColor: series.map((p) => (p.ts ? '#fff' : '#1f6feb')),
        pointBorderWidth: series.map((p) => (p.ts ? 2 : 0)),
        pointHoverRadius: series.map((p) => (p.ts ? 8 : 6)),
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => items.length
              ? `${coord.kind || 'coordinate'} = ` +
                `${Number(items[0].parsed.x).toFixed(3)} ${unit}`
              : '',
            label: (item) => {
              const v = Number(item.parsed.y);
              const tag = series[item.dataIndex]?.ts ? ' (transition state)' : '';
              return `${v.toFixed(2)} kcal/mol${tag}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: `${coord.kind || 'reaction coordinate'} (${unit})`,
          },
        },
        y: {
          title: { display: true, text: 'Energy (kcal/mol)' },
        },
      },
    },
    plugins: [guides],
  });
}

function renderIrc(r) {
  const block = $('#ircBlock');
  if (!block) return;
  const irc = r.irc || {};
  const br = irc.branches || {};
  if (!br.forward && !br.backward) {
    block.classList.add('hidden');
    if (irc.skipped) $('#ircNote').textContent = irc.skipped;
    return;
  }
  block.classList.remove('hidden');

  const fwd = br.forward || {};
  const bwd = br.backward || {};
  const drops = [fwd.total_drop_kcal, bwd.total_drop_kcal]
    .filter((v) => typeof v === 'number');
  const bits = [`${irc.steps} steps per branch at `
    + `${Number(irc.step_size).toFixed(2)} amu\u00bd bohr`];
  if (drops.length) {
    bits.push(`each branch drops ${Math.abs(drops[0]).toFixed(2)} kcal/mol`);
  }
  if (irc.preconditioned) bits.push('Hessian-preconditioned');
  $('#ircSub').innerHTML = bits.map(escapeHtml).join(' &middot; ');

  const ctx = $('#ircChart').getContext('2d');
  if (state.charts.irc) state.charts.irc.destroy();

  // Energy measured from the *first* IRC point on each branch, so both
  // branches start at 0 and the two curves are directly comparable.
  const mk = (b, label, colour) => {
    const pts = b.points || [];
    if (!pts.length) return null;
    const e0 = Number(pts[0].energy_hartree);
    return {
      label,
      data: pts.map((p, i) => ({
        x: i,
        y: (Number(p.energy_hartree) - e0) * 627.5094740631,
        coord: p.coord,
      })),
      borderColor: colour,
      backgroundColor: colour,
      borderWidth: 2,
      tension: 0,
      fill: false,
      pointRadius: 3,
      pointHoverRadius: 6,
    };
  };
  const sets = [mk(bwd, 'backward', '#8b5cf6'),
    mk(fwd, 'forward', '#d6604d')].filter(Boolean);

  state.charts.irc = new Chart(ctx, {
    type: 'line',
    data: { datasets: sets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: {
        legend: { display: true, labels: { boxWidth: 10, font: { size: 10 } } },
        tooltip: {
          callbacks: {
            label: (item) => {
              const p = sets[item.datasetIndex].data[item.dataIndex];
              return `${Number(item.parsed.y).toFixed(3)} kcal/mol` +
                (p.coord === null || p.coord === undefined
                  ? '' : `  (${Number(p.coord).toFixed(1)}\u00b0)`);
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: 'IRC step' } },
        y: {
          title: {
            display: true,
            text: 'Energy from the first IRC point (kcal/mol)',
          },
        },
      },
    },
  });

  const notes = [];
  if (irc.note) notes.push(irc.note);
  if (!fwd.descends_monotonically || !bwd.descends_monotonically) {
    notes.push('At least one branch does not descend at every step, so the ' +
      'step size is too large for this surface and this is not a reliable IRC.');
  }
  notes.push('Each point is a converged SCF with an analytic gradient on a ' +
    'dense quadrature grid; the rigid-body directions are projected out at ' +
    'every step so the molecule cannot drift.');
  $('#ircNote').textContent = notes.join(' ');
}

function ircToCsv() {
  const r = state.reaction;
  const br = (r && r.irc && r.irc.branches) || {};
  if (!br.forward && !br.backward) return;
  const q = (v) => {
    const t = v === null || v === undefined ? '' : String(v);
    return /[",\n]/.test(t) ? '"' + t.replace(/"/g, '""') + '"' : t;
  };
  const rows = ['branch,step,coordinate,energy_kcal_from_first,grad_rms'];
  ['forward', 'backward'].forEach((name) => {
    const b = br[name];
    if (!b) return;
    const pts = b.points || [];
    if (!pts.length) return;
    const e0 = Number(pts[0].energy_hartree);
    pts.forEach((p, i) => rows.push([
      name, i,
      p.coord === null || p.coord === undefined ? '' : Number(p.coord).toFixed(4),
      ((Number(p.energy_hartree) - e0) * 627.5094740631).toFixed(4),
      Number(p.grad_rms).toExponential(3),
    ].map(q).join(',')));
  });
  const meta = `# ${r.functional_label || r.functional} / ` +
    `${r.basis_label || r.basis}` +
    `\n# irc_steps,${r.irc.steps}` +
    `\n# irc_step_size_amu_half_bohr,${r.irc.step_size}` +
    `\n# preconditioned,${r.irc.preconditioned ? 'yes' : 'no'}`;
  saveDataUrl('data:text/csv;charset=utf-8,' +
    encodeURIComponent(meta + '\n' + rows.join('\n')), 'irc.csv');
}

function renderReactionTable(r) {
  const table = $('#reactionTable');
  if (!table) return;
  let html = '<thead><tr><th>#</th>' +
    `<th>${escapeHtml((r.coordinate || {}).kind || 'coordinate')}</th>` +
    '<th>Energy (kcal/mol)</th><th>SCF steps</th></tr></thead><tbody>';
  const tsCoord = r.ts ? Number(r.ts.coord) : null;
  r.points.forEach((p, i) => {
    const isTs = tsCoord !== null && Math.abs(Number(p.coord) - tsCoord) < 1e-9;
    html += `<tr${isTs ? ' class="row-ts"' : ''}>`
      + `<td>${i + 1}</td>`
      + `<td>${Number(p.coord).toFixed(4)}${isTs ? ' \u25c0 max' : ''}</td>`
      + `<td>${Number(p.relative_kcal).toFixed(3)}</td>`
      + `<td>${p.steps === null || p.steps === undefined ? '\u2014' : p.steps}</td>`
      + '</tr>';
  });
  table.innerHTML = html + '</tbody>';
}

function reactionToCsv() {
  const r = state.reaction;
  if (!r) return;
  const coord = r.coordinate || {};
  const q = (v) => {
    const t = v === null || v === undefined ? '' : String(v);
    return /[",\n]/.test(t) ? '"' + t.replace(/"/g, '""') + '"' : t;
  };
  const rows = [`# ${coord.kind || 'coordinate'} (${coord.unit || ''}),` +
    'energy_kcal_per_mol,kind'];
  (r.points || []).forEach((p) => rows.push(
    [Number(p.coord).toFixed(5), Number(p.relative_kcal).toFixed(4), 'scan']
      .map(q).join(',')));
  if (r.ts && r.ts.coord !== null && r.ts.coord !== undefined) {
    rows.push([Number(r.ts.coord).toFixed(5),
      Number(r.ts.relative_kcal).toFixed(4), 'transition_state'].map(q).join(','));
  }
  rows.push(['reactant', Number(r.reactant?.relative_kcal ?? 0).toFixed(4),
    'relaxed_endpoint'].map(q).join(','));
  rows.push(['product', Number(r.product?.relative_kcal ?? 0).toFixed(4),
    'relaxed_endpoint'].map(q).join(','));
  const meta = `# ${r.functional_label || r.functional} / ` +
    `${r.basis_label || r.basis}` +
    `\n# barrier_kcal,${r.barrier_kcal === null ? '' : r.barrier_kcal}` +
    `\n# reaction_kcal,${r.reaction_kcal}` +
    `\n# transition_state_verified,` +
    `${(r.verification || {}).is_transition_state ? 'yes' : 'no'}` +
    `\n# n_imaginary,${(r.verification || {}).n_imaginary ?? ''}`;
  saveDataUrl('data:text/csv;charset=utf-8,' +
    encodeURIComponent(meta + '\n' + rows.join('\n')), 'reaction-path.csv');
}

/* ---- real-space fields: ELF, Laplacian, spin density, difference ------ */
const FIELD_LABELS = {
  elf: 'Electron localisation function (ELF)',
  laplacian: 'Laplacian of the density',
  spin: 'Spin density',
  density: 'Electron density',
  gradient: 'Density gradient magnitude',
  difference: 'Density difference',
};

// viridis, sampled: perceptually uniform and safe for colour-blind readers,
// unlike the rainbow ramps most plotting defaults still use.
const VIRIDIS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
  [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
  [180, 222, 44], [253, 231, 37],
];
const DIVERGING = [
  [5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222],
  [247, 247, 247], [244, 165, 130], [214, 96, 77], [178, 24, 43],
  [103, 0, 31],
];

function rampColor(stops, t) {
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},` +
         `${Math.round(a[1] + (b[1] - a[1]) * f)},` +
         `${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}

// Marching squares.  bit order: top-left 8, top-right 4, bottom-right 2,
// bottom-left 1.  Each entry lists pairs of cell edges the contour crosses.
const MS_CASES = {
  1: [['left', 'bottom']], 2: [['bottom', 'right']],
  3: [['left', 'right']], 4: [['top', 'right']],
  5: [['left', 'top'], ['bottom', 'right']], 6: [['top', 'bottom']],
  7: [['left', 'top']], 8: [['left', 'top']],
  9: [['top', 'bottom']], 10: [['top', 'right'], ['left', 'bottom']],
  11: [['top', 'right']], 12: [['left', 'right']],
  13: [['bottom', 'right']], 14: [['left', 'bottom']],
};

function contourPaths(grid, n, level) {
  const paths = [];
  const lerp = (a, b) => (b === a ? 0.5 : (level - a) / (b - a));
  for (let r = 0; r < n - 1; r++) {
    for (let c = 0; c < n - 1; c++) {
      const v00 = grid[r][c], v10 = grid[r][c + 1];
      const v01 = grid[r + 1][c], v11 = grid[r + 1][c + 1];
      if (![v00, v10, v01, v11].every(Number.isFinite)) continue;
      const idx = (v00 > level ? 8 : 0) | (v10 > level ? 4 : 0) |
                  (v11 > level ? 2 : 0) | (v01 > level ? 1 : 0);
      const cases = MS_CASES[idx];
      if (!cases) continue;
      const pt = {
        top: () => [c + lerp(v00, v10), r],
        right: () => [c + 1, r + lerp(v10, v11)],
        bottom: () => [c + lerp(v01, v11), r + 1],
        left: () => [c, r + lerp(v00, v01)],
      };
      for (const [ea, eb] of cases) {
        const a = pt[ea](), b = pt[eb]();
        if (a && b) paths.push([a, b]);
      }
    }
  }
  return paths;
}

function renderField(job) {
  const f = job.result;
  if (!f || !f.values) return;
  state.field = f;

  $('#emptyFields')?.classList.add('hidden');
  $('#fieldContent')?.classList.remove('hidden');

  const m = f.molecule || {};
  $('#fieldTitle').textContent = `${f.label || f.kind} of ${m.name || m.formula || 'the molecule'}`;
  const bits = [`${f.functional_label || f.functional} / ${f.basis_label || f.basis}`];
  if (f.converged === false) bits.push('SCF NOT CONVERGED');
  if (f.stats) {
    bits.push(`range ${f.stats.min.toExponential(2)} … ${f.stats.max.toExponential(2)}`);
  }
  if (f.units) bits.push(f.units);
  $('#fieldSub').innerHTML = bits.map(escapeHtml).join(' &middot; ');

  const sel = $('#fieldKind');
  sel.innerHTML = Object.keys(FIELD_LABELS)
    .map((k) => `<option value="${k}">${escapeHtml(FIELD_LABELS[k])}</option>`).join('');
  sel.value = f.kind || 'elf';
  $('#fieldPlane').value = f.plane || 'xy';
  $('#fieldNote').textContent = f.note || '';

  drawField();

  if (job.explanation) addMessage('assistant', markdown(job.explanation));
  switchTab('fields');
}

function fieldScale() {
  const f = state.field;
  const kind = f.kind || 'elf';
  const levels = f.levels || [];
  if (kind === 'elf') return { lo: 0, hi: 1, signed: false };
  if (levels.length) {
    const lo = levels[0], hi = levels[levels.length - 1];
    return { lo: Math.min(lo, hi), hi: Math.max(lo, hi),
             signed: lo < 0 && hi > 0 };
  }
  const s = f.stats || {};
  return { lo: s.min || 0, hi: s.max || 1, signed: (s.min || 0) < 0 };
}

function drawField() {
  const f = state.field;
  if (!f) return;
  const cv = $('#fieldCanvas');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const grid = f.values;
  const n = f.n || grid.length;
  const { lo, hi, signed } = fieldScale();
  const span = (hi - lo) || 1;
  const stops = signed ? DIVERGING : VIRIDIS;

  ctx.clearRect(0, 0, W, H);
  const cw = W / n, ch = H / n;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = grid[r] && grid[r][c];
      if (v === undefined || v === null || !Number.isFinite(v)) continue;
      const t = signed ? (v - lo) / span : (v - lo) / span;
      ctx.fillStyle = rampColor(stops, t);
      // row 0 is v_min; canvas y grows downward, so flip to make v increase up
      ctx.fillRect(c * cw, H - (r + 1) * ch, cw + 1, ch + 1);
    }
  }

  // ---- contour lines ---------------------------------------------------
  if ($('#fieldContours')?.checked && f.levels && f.levels.length) {
    ctx.lineWidth = 1;
    for (const lv of f.levels) {
      if (lv <= lo || lv >= hi) continue;
      const segs = contourPaths(grid, n, lv);
      // zero contour on a signed field is the one a reader looks for first
      const isZero = signed && Math.abs(lv) < 1e-12;
      ctx.strokeStyle = isZero ? 'rgba(20,20,20,.85)'
                               : 'rgba(255,255,255,.35)';
      ctx.lineWidth = isZero ? 1.6 : 0.8;
      ctx.beginPath();
      for (const [a, b] of segs) {
        ctx.moveTo(a[0] * cw, H - a[1] * ch);
        ctx.lineTo(b[0] * cw, H - b[1] * ch);
      }
      ctx.stroke();
    }
  }

  // ---- atoms projected into the plane ----------------------------------
  const u0 = (f.u_range || [0, 1])[0], u1 = (f.u_range || [0, 1])[1];
  const v0 = (f.v_range || [0, 1])[0], v1 = (f.v_range || [0, 1])[1];
  const su = (u1 - u0) || 1, sv = (v1 - v0) || 1;
  (f.atoms || []).forEach((a) => {
    const x = ((a.u - u0) / su) * W;
    const y = H - ((a.v - v0) / sv) * H;
    ctx.beginPath();
    ctx.arc(x, y, 4.2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,.92)';
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = '#16202e';
    ctx.stroke();
    if (a.symbol !== 'H') {
      ctx.fillStyle = '#16202e';
      ctx.font = '600 10px var(--sans, sans-serif)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(a.symbol, x, y);
    }
  });

  // ---- colour bar ------------------------------------------------------
  const bar = $('#fieldBar');
  if (bar) {
    const bc = bar.getContext('2d');
    for (let x = 0; x < bar.width; x++) {
      bc.fillStyle = rampColor(stops, x / (bar.width - 1));
      bc.fillRect(x, 0, 1, bar.height);
    }
  }
  const lab = $('#fieldScaleLabels');
  if (lab) {
    const fmt = (v) => (Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3))
      ? v.toExponential(1) : v.toFixed(Math.abs(hi - lo) < 2 ? 3 : 2);
    lab.textContent = `${fmt(lo)}   ${signed ? fmt((lo + hi) / 2) + '   ' : ''}${fmt(hi)}` +
      (f.units ? `  ${f.units}` : '');
  }
}

async function fieldRecompute() {
  const f = state.field;
  if (!f) return;
  const kind = $('#fieldKind').value;
  const plane = $('#fieldPlane').value;
  if (kind === f.kind && plane === f.plane) { drawField(); return; }
  const m = f.molecule || {};
  const spec = m.smiles || m.name || '';
  if (!spec) return;
  try {
    const res = await api('/api/field', {
      method: 'POST',
      body: JSON.stringify({
        molecule: spec, kind: kind, plane: plane, n: f.n || 80,
        functional: f.functional || 'b3lyp', basis: f.basis || '6-31g*',
      }),
    });
    state.activeJob = res.job_id;
    beginPolling(res.job_id);
  } catch (e) {
    addMessage('assistant', `<div class="error-box">${escapeHtml(e.message)}</div>`);
  }
}

function fieldToPng() {
  const cv = $('#fieldCanvas');
  if (!cv || !state.field) return;
  saveDataUrl(cv.toDataURL('image/png'),
    `field_${state.field.kind}_${state.field.plane}.png`);
}

function fieldToCsv() {
  const f = state.field;
  if (!f) return;
  const n = f.n || f.values.length;
  const u0 = (f.u_range || [0, 1])[0], u1 = (f.u_range || [0, 1])[1];
  const v0 = (f.v_range || [0, 1])[0], v1 = (f.v_range || [0, 1])[1];
  const du = (u1 - u0) / (n - 1), dv = (v1 - v0) / (n - 1);
  const rows = [`# ${f.label} (${f.units || 'a.u.'}) on the ${f.plane} plane, ` +
                `${n}x${n} points`, `# ${f.functional_label || f.functional} / ` +
                `${f.basis_label || f.basis}`, 'u_A,v_A,value'];
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      rows.push(`${(u0 + c * du).toFixed(5)},${(v0 + r * dv).toFixed(5)},` +
                `${f.values[r][c]}`);
    }
  }
  saveDataUrl('data:text/csv;charset=utf-8,' +
    encodeURIComponent(rows.join('\n')), `field_${f.kind}.csv`);
}

/* ---- automatic molecular design -------------------------------------- */
const DESIGN_ORIGIN_LABEL = {
  model: 'model',
  seed: 'library',
  mutation: 'mutated',
  assembly: 'assembled',
  analogue: 'analogue of the named molecule',
};

function renderDesign(job) {
  const d = job.result;
  if (!d || !d.candidates) return;
  state.design = d;

  const box = $('#designContent');
  box.classList.remove('hidden');
  $('#emptyDesign')?.classList.add('hidden');

  const brief = d.brief || {};
  $('#designGoal').textContent = brief.goal || brief.text || 'Design run';
  const sub = [];
  if (d.generated) sub.push(`${d.generated} proposed`);
  if (d.rejected) sub.push(`${d.rejected} rejected`);
  if (d.dft_ran) sub.push(`${d.dft_ran} calculated`);
  if (d.dft_cached) sub.push(`${d.dft_cached} reused`);
  if (d.model_used) sub.push(`model ${escapeHtml(String(d.model_used))}`);
  if (d.seconds) sub.push(`${d.seconds.toFixed(1)} s`);
  $('#designSub').innerHTML = sub.join(' &middot; ');

  // ---- what the run was aiming for -----------------------------------
  const tg = $('#designTargets');
  tg.innerHTML = '';
  (d.requirements || []).forEach((s) => {
    const unmet = (d.unmet_requirements || []).includes(s);
    const chip = el('span', 'design-target' + (unmet ? ' unmet' : ''));
    chip.innerHTML = `<em>must contain</em> <code>${escapeHtml(s)}</code>`
      + (unmet ? ' <span>not built in this run</span>' : '');
    tg.appendChild(chip);
  });
  (d.targets || []).forEach((t) => {
    const chip = el('span', 'design-target');
    const aim = t.aim === 'max' ? 'maximise' : t.aim === 'min' ? 'minimise' : 'hit';
    const win = [];
    if (t.low !== null && t.low !== undefined) win.push(`&ge;${t.low}`);
    if (t.value !== null && t.value !== undefined) win.push(`~${t.value}`);
    if (t.high !== null && t.high !== undefined) win.push(`&le;${t.high}`);
    chip.innerHTML =
      `<em>${escapeHtml(t.label || t.key)}</em> ${aim}` +
      (win.length ? ` <span>${win.join(' ')}</span>` : '');
    tg.appendChild(chip);
  });

  // ---- origin breakdown and any engine notes -------------------------
  const sum = $('#designSummary');
  const bits = [];
  const origins = d.origins || {};
  const total = Object.values(origins).reduce((a, b) => a + b, 0);
  if (total) {
    bits.push(`<span>${total} candidates kept: ` + Object.keys(origins)
      .sort((a, b) => origins[b] - origins[a])
      .map((k) => `${origins[k]} ${escapeHtml(DESIGN_ORIGIN_LABEL[k] || k)}`)
      .join(', ') + '</span>');
  }
  const reasons = d.rejection_reasons || {};
  const topReasons = Object.keys(reasons).slice(0, 3);
  if (topReasons.length) {
    bits.push('<span class="muted">dropped mostly for: '
      + topReasons.map((r) => `${escapeHtml(r)} (${reasons[r]})`).join('; ')
      + '</span>');
  }
  const checks = d.quality_checks || {};
  const on = Object.keys(checks).filter((k) => checks[k]);
  if (on.length) {
    const names = {
      sa_score: 'synthetic accessibility',
      qed: 'QED',
      alerts: 'PAINS / BRENK / NIH alerts',
    };
    bits.push('<span class="muted">quality filters: '
      + on.map((k) => escapeHtml(names[k] || k)).join(', ') + '</span>');
  }
  (d.model_notes || []).forEach((n) => {
    bits.push(`<span class="muted">${escapeHtml(n)}</span>`);
  });
  sum.innerHTML = bits.join('');

  // ---- the cards -------------------------------------------------------
  const list = $('#designList');
  list.innerHTML = '';
  d.candidates.forEach((c, i) => list.appendChild(designCard(c, i)));

  const notes = [];
  const relaxed = d.candidates.some((c) => c.dft && c.dft.optimized);
  if (!d.dft_ran && !d.dft_cached) {
    notes.push('No electronic-structure calculation was part of this run: '
      + 'every number above is a structure-based descriptor from RDKit.');
  } else {
    notes.push('Descriptors come from RDKit; HOMO, LUMO, gap and dipole are '
      + 'measured at the level of theory shown on each card.');
    notes.push(relaxed
      ? 'The lead geometries were relaxed at that same level of theory, so '
        + 'those numbers come from a real minimum.'
      : 'The geometry is the RDKit ETKDGv3/MMFF94 conformer, written A//B for '
        + 'a single point at A on a geometry optimised at B. Ask to "optimise '
        + 'the geometry" to have the leads relaxed at the DFT level first.');
  }
  $('#designNote').textContent = notes.join(' ');

  if (job.explanation) {
    addMessage('assistant', markdown(job.explanation));
  }
  switchTab('design');
}

function designCard(c, index) {
  const card = el('div', 'design-card');
  const head = el('div', 'design-card-head');

  const rank = el('span', 'design-rank', String(index + 1));
  head.appendChild(rank);

  const title = el('div', 'design-card-title');
  title.innerHTML =
    `<strong>${escapeHtml(c.name || c.smiles)}</strong>` +
    `<code>${escapeHtml(c.smiles)}</code>`;
  head.appendChild(title);

  const score = el('span', 'design-score');
  if (c.scoring.scored === false) {
    // A gap-only brief has nothing to rank on until the SCF has run.  Showing
    // "0 score" there reads as "fails every criterion", which is a different
    // and untrue statement.
    score.innerHTML = '<em>&mdash;</em> not scored yet';
    score.title = c.scoring.unscored_reason || '';
  } else {
    score.innerHTML = `<em>${(c.scoring.score * 100).toFixed(0)}</em> score`;
  }
  head.appendChild(score);
  card.appendChild(head);

  const badges = el('div', 'design-badges');
  badges.appendChild(el('span', 'badge origin-' + (c.origin || 'seed'),
    DESIGN_ORIGIN_LABEL[c.origin] || c.origin || ''));
  const desc = c.descriptors || {};
  if (desc.formula) badges.appendChild(el('span', 'badge', escapeHtml(desc.formula)));
  if (desc.heavy_atoms) badges.appendChild(el('span', 'badge', desc.heavy_atoms + ' heavy atoms'));
  if (c.scoring.scored === false) {
    badges.appendChild(el('span', 'badge warn',
      c.scoring.unscored_reason || 'not scored yet'));
  }
  if (c.qualified === false) {
    badges.appendChild(el('span', 'badge warn', 'missing a required group'));
  }
  // a structural alert is a liability, not a disqualification, so it is
  // shown rather than hidden -- but it is never left out either
  Object.keys(desc.alerts || {}).forEach((set) => {
    const hits = desc.alerts[set];
    if (!hits || !hits.length) return;
    badges.appendChild(el('span', 'badge warn',
      `${set}: ${hits[0]}${hits.length > 1 ? ` +${hits.length - 1}` : ''}`));
  });
  const blocking = c.scoring.blocking || [];
  if (blocking.length) {
    badges.appendChild(el('span', 'badge warn',
      'fails ' + blocking.join(', ')));
  }
  card.appendChild(badges);

  // ---- descriptors ----------------------------------------------------
  const vals = desc.values || {};
  const keys = ['molwt', 'logp', 'tpsa', 'hbd', 'hba', 'rotb', 'arom_rings',
    'fsp3', 'sa', 'qed'];
  const grid = el('div', 'design-desc');
  keys.forEach((k) => {
    if (vals[k] === null || vals[k] === undefined) return;
    const cell = el('div', 'design-desc-cell');
    cell.innerHTML = `<span>${escapeHtml(k)}</span><strong>${vals[k]}</strong>`;
    grid.appendChild(cell);
  });
  if (grid.children.length) card.appendChild(grid);

  // ---- measured electronic structure ----------------------------------
  const dft = c.dft;
  if (dft) {
    const box = el('div', 'design-dft');
    const level = `${dft.functional_label || dft.functional || ''}`
      + (dft.basis_label ? ` / ${dft.basis_label}` : '');
    // the level of theory alone is not reproducible: A//B means a single
    // point at A on a geometry optimised at B
    const geom = dft.optimized ? ''
      : ` // ${escapeHtml(String(dft.geometry_source || 'MMFF94').split(' (')[0])}`;
    box.innerHTML =
      `<div class="design-dft-head">measured${level ? ' at ' + escapeHtml(level) : ''}`
      + `${geom}`
      + `${dft.scf_seconds ? ` &middot; ${dft.scf_seconds.toFixed(1)} s` : ''}`
      + `${dft.cached ? ' &middot; reused' : ''}`
      + `${dft.optimized && dft.opt_seconds
          ? ` &middot; relaxed in ${(+dft.opt_seconds).toFixed(0)} s` : ''}`
      + '</div>' +
      '<div class="design-desc">' +
      ['gap_ev:gap (eV):3', 'homo_ev:HOMO (eV):3', 'lumo_ev:LUMO (eV):3',
       'dipole:dipole (D):2']
        .map((spec) => spec.split(':'))
        .map(([k, label, dig]) => {
          const v = dft[k];
          if (v === null || v === undefined) return '';
          return `<div class="design-desc-cell"><span>${label}</span>` +
                 `<strong>${(+v).toFixed(+dig)}</strong></div>`;
        }).join('') +
      '</div>';
    card.appendChild(box);
  } else if (c.dft_error) {
    card.appendChild(el('div', 'design-dft warn', c.dft_error));
  }

  // ---- per-criterion desirability bars --------------------------------
  const bars = el('div', 'design-bars');
  (c.scoring.targets || []).forEach((t) => {
    const row = el('div', 'design-bar-row');
    const pct = t.desirability === null || t.desirability === undefined
      ? 0 : Math.round(t.desirability * 100);
    row.innerHTML =
      `<span class="design-bar-label">${escapeHtml(t.label || t.key)}</span>` +
      `<span class="design-bar"><i style="width:${pct}%"></i></span>` +
      `<span class="design-bar-val">` +
      (t.desirability === null || t.desirability === undefined
        ? 'not measured' : `${pct}%`) +
      `</span>` +
      (t.value !== null && t.value !== undefined
        ? `<span class="design-bar-raw">${t.value}${escapeHtml(t.unit || '')}</span>`
        : '');
    bars.appendChild(row);
  });
  card.appendChild(bars);

  if (c.rationale) {
    const p = el('p', 'design-rationale', c.rationale);
    card.appendChild(p);
  }
  if (c.discussion) {
    card.appendChild(el('p', 'design-discussion', c.discussion));
  }

  // ---- actions ---------------------------------------------------------
  const actions = el('div', 'design-actions');
  const view = el('button', 'btn tiny ghost', 'Show structure');
  view.type = 'button';
  view.addEventListener('click', () => designShow(c.smiles));
  actions.appendChild(view);

  const run = el('button', 'btn tiny', 'Calculate');
  run.type = 'button';
  run.addEventListener('click', () => designCalculate(c.smiles, c.name));
  actions.appendChild(run);
  card.appendChild(actions);

  return card;
}

async function designShow(smiles) {
  try {
    const mol = await api('/api/molecule/resolve', {
      method: 'POST',
      body: JSON.stringify({ spec: smiles, kind: 'smiles' }),
    });
    showMolecule(mol);
    renderMoleculeOnly(mol);
    switchTab('geometry');
  } catch (e) {
    addMessage('assistant',
      `<div class="error-box">${escapeHtml(e.message)}</div>`);
  }
}

async function designCalculate(smiles, name) {
  try {
    const body = {
      molecule: smiles,
      kind: 'single_point',
      functional: $('#selFunctional')?.value || 'b3lyp',
      basis: $('#selBasis')?.value || '6-31g*',
      name: name || '',
    };
    const res = await api('/api/job', {
      method: 'POST',
      body: JSON.stringify(body),
    });
    state.activeJob = res.job_id;
    beginPolling(res.job_id);
  } catch (e) {
    addMessage('assistant',
      `<div class="error-box">${escapeHtml(e.message)}</div>`);
  }
}

function designToCsv() {
  const d = state.design;
  if (!d || !d.candidates || !d.candidates.length) return;
  const rows = [['rank', 'name', 'smiles', 'origin', 'scored', 'score',
                 'formula',
                 'molwt', 'logp', 'tpsa', 'hbd', 'hba', 'rotb',
                 'SA_score', 'QED', 'alerts', 'fails', 'geometry',
                 'Gap_eV', 'HOMO_eV', 'LUMO_eV', 'Dipole_D', 'rationale']];
  d.candidates.forEach((c, i) => {
    const v = (c.descriptors && c.descriptors.values) || {};
    const f = c.dft || {};
    const alerts = Object.keys(c.descriptors && c.descriptors.alerts || {})
      .map((k) => `${k}:${c.descriptors.alerts[k].join('|')}`).join('; ');
    const geometry = f.optimized ? 'DFT-optimised'
      : (f.geometry_source ? String(f.geometry_source).split(' (')[0] : '');
    // An unscored candidate gets a blank, not a zero: a zero in a spreadsheet
    // is a value, and "no value exists yet" is a different thing.
    const scored = c.scoring.scored !== false;
    rows.push([
      i + 1, c.name || '', c.smiles, c.origin || '',
      scored ? 'yes' : 'no', scored ? c.scoring.score : '',
      (c.descriptors && c.descriptors.formula) || '',
      v.molwt, v.logp, v.tpsa, v.hbd, v.hba, v.rotb,
      v.sa, v.qed, alerts,
      (c.scoring.blocking || []).join('; '), geometry,
      f.gap_ev, f.homo_ev, f.lumo_ev, f.dipole,
      (c.rationale || '').replace(/"/g, "'"),
    ]);
  });
  const text = rows.map((r) => r.map((x) => {
    const s = x === null || x === undefined ? '' : String(x);
    return /[",\n]/.test(s) ? `"${s}"` : s;
  }).join(',')).join('\n');
  const url = URL.createObjectURL(new Blob([text],
    { type: 'text/csv;charset=utf-8' }));
  saveDataUrl(url, `design-${d.id || 'run'}.csv`);
}

function applyNCISurface() {
  const d = state.nci;
  if (!d || !state.viewer) return;
  const key = $('#nciSelect').value;
  const it = (d.surfaces || []).find((x) => x.key === key);
  if (!it) return;

  const status = $('#nciStatus');
  const url = `${d.url_base}/${it.file}`;
  status.className = 'surf-status busy';
  status.textContent = 'loading grid…';

  const cached = state.cubeCache && state.cubeCache.get(url);
  const work = cached
    ? Promise.resolve(cached)
    : fetch(url).then((resp) => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.text();
      }).then((text) => {
        state.cubeCache.set(url, text);
        return text;
      });

  work.then((text) => {
    const viewer = state.viewer;
    viewer.removeAllShapes();
    viewer.removeAllSurfaces();
    // Each class lives in its own grid, so the isosurface is drawn in the
    // class colour directly instead of trying to recolour a single grid.
    viewer.addVolumetricData(text, 'cube', {
      isoval: +$('#nciIso').value || 0.5,
      color: it.colour,
      opacity: +$('#nciOpacity').value || 0.85,
      smoothness: 2,
    });
    viewer.render();
    status.className = 'surf-status';
    status.textContent =
      `${it.label} · RDG ${(+$('#nciIso').value).toFixed(2)} a.u.`;
  }).catch((err) => {
    status.className = 'surf-status error';
    status.textContent = `could not load ${it.file}: ${err.message}`;
  });
}

function renderGeometry(geo) {
  $('#geomContent')?.classList.remove('hidden');
  const p = document.querySelector('[data-pane="geometry"] .empty-state');
  if (p) p.classList.add('hidden');

  const fill = (table, rows, unit) => {
    if (!rows || !rows.length) {
      table.innerHTML = '<thead><tr><th>No data</th></tr></thead><tbody></tbody>';
      return;
    }
    let html = `<thead><tr><th>${unit}</th><th>Value</th></tr></thead><tbody>`;
    rows.forEach((r) => {
      html += `<tr><td>${escapeHtml(r.label)}</td><td>${r.value.toFixed(4)}</td></tr>`;
    });
    table.innerHTML = html + '</tbody>';
  };

  fill($('#bondTable'), geo.bonds, 'Bond');
  fill($('#angleTable'), geo.angles, 'Angle');
  fill($('#dihTable'), geo.dihedrals, 'Dihedral');
}

function renderGeometryFromBonds(mol) {
  const bonds = (mol.bonds || []).map((b) => ({
    label: `${mol.atoms[b.i].symbol}${b.i + 1}-${mol.atoms[b.j].symbol}${b.j + 1}`,
    value: b.length,
  })).sort((a, b) => a.value - b.value);
  renderGeometry({ bonds: bonds, angles: [], dihedrals: [] });
}

function renderMoleculeOnly(mol) {
  if (state.lastResult) return; // a real result will overwrite this
  renderGeometryFromBonds(mol);
}

/* ---- compare ----------------------------------------------------------- */
function renderCompare(job) {
  const { a, b } = job.result;
  $('#emptyProps')?.classList.add('hidden');
  $('#propContent')?.classList.remove('hidden');
  $('#statGrid').innerHTML = '';
  $('#dipoleBlock').innerHTML = '';
  $('#chargeBlock').innerHTML = '';

  const table = el('div', 'card');
  const rows = [
    ['', 'Molecule A', 'Molecule B'],
    ['Name', a.molecule.name || a.molecule.formula, b.molecule.name || b.molecule.formula],
    ['Formula', a.molecule.formula, b.molecule.formula],
    ['Atoms', a.molecule.natoms, b.molecule.natoms],
    ['Method', a.functional_label + ' / ' + a.basis_label, b.functional_label + ' / ' + b.basis_label],
    ['Total energy (Ha)', fmt(a.energy_hartree, 6), fmt(b.energy_hartree, 6)],
    ['HOMO (eV)', fmt(a.homo_ev, 3), fmt(b.homo_ev, 3)],
    ['LUMO (eV)', fmt(a.lumo_ev, 3), fmt(b.lumo_ev, 3)],
    ['Gap (eV)', fmt(a.gap_ev, 3), fmt(b.gap_ev, 3)],
    ['Dipole (D)', fmt(a.dipole.magnitude, 3), fmt(b.dipole.magnitude, 3)],
    ['SCF', a.converged ? 'converged' : 'no', b.converged ? 'converged' : 'no'],
  ];
  let html = '<h3>Side-by-side comparison</h3><div class="table-wrap"><table>';
  html += '<thead><tr><th>Property</th><th>' +
    escapeHtml(a.molecule.name || a.molecule.formula) + '</th><th>' +
    escapeHtml(b.molecule.name || b.molecule.formula) + '</th></tr></thead><tbody>';
  rows.slice(1).forEach((r) => {
    html += `<tr><td>${escapeHtml(String(r[0]))}</td><td>${escapeHtml(String(r[1]))}</td>` +
      `<td>${escapeHtml(String(r[2]))}</td></tr>`;
  });
  html += '</tbody></table></div>';
  table.innerHTML = html;
  $('#propContent').insertBefore(table, $('#propContent').firstChild.nextSibling);

  showMolecule(a.molecule);
  switchTab('properties');

  if (job.explanation) {
    appendNarrative(job, job.explanation);
    addMessage('assistant', markdown(job.explanation));
    switchTab('narrative');
  }
}

/* ---- narrative --------------------------------------------------------- */
function appendNarrative(job, text) {
  const box = $('#narrContent');
  const p = document.querySelector('[data-pane="narrative"] .empty-state');
  if (p) p.classList.add('hidden');

  const item = el('div', 'narrative-item');
  const title = job.result && job.result.molecule
    ? (job.result.molecule.name || job.result.molecule.formula)
    : job.kind;
  item.innerHTML = `<h4>${escapeHtml(title)} &middot; ${escapeHtml(job.kind.replace(/_/g, ' '))}</h4>` +
    markdown(text);
  box.insertBefore(item, box.firstChild);
}

/* ------------------------------------------------------------------ jobs list */
async function refreshJobs(quiet) {
  try {
    const data = await api('/api/jobs');
    state.jobs = data.jobs || [];
    renderJobs();
  } catch (e) {
    if (!quiet) {
      const box = $('#jobsList');
      if (box) box.innerHTML = `<p class="muted">Could not load jobs: ${escapeHtml(e.message)}</p>`;
    }
  }
}

function renderJobs() {
  const box = $('#jobsList');
  if (!state.jobs.length) {
    box.innerHTML = '<p class="muted">No jobs yet.</p>';
    return;
  }
  box.innerHTML = '';
  state.jobs.forEach((j) => {
    const item = el('div', `job-item ${j.status}`);
    const label = j.payload_name || j.kind.replace(/_/g, ' ');
    item.innerHTML =
      `<span class="st"></span>` +
      `<span class="title">${escapeHtml(label)}</span>` +
      `<span class="muted">${j.status}</span>` +
      `<span class="tm">${j.elapsed}s</span>`;
    item.addEventListener('click', () => focusJob(j));
    box.appendChild(item);
  });
}

async function focusJob(j) {
  if (j.status === 'running' || j.status === 'queued') {
    state.activeJob = j.id;
    beginPolling(j.id);
    return;
  }
  if (j.status === 'failed') {
    addMessage('assistant', `<div class="error-box">${escapeHtml(j.message || 'failed')}</div>`);
    return;
  }
  try {
    const full = await api(`/api/job/${j.id}`);
    handleCompleted(full);
  } catch (e) {
    addMessage('assistant', `<div class="error-box">${escapeHtml(e.message)}</div>`);
  }
}

/* ------------------------------------------------------------------ library */
const CATEGORY_LABEL = {
  basics: 'Basics', aromatics: 'Aromatics', heterocycles: 'Heterocycles',
  carbonyls: 'Carbonyls', alcohols: 'Alcohols', acids: 'Acids',
  biomolecules: 'Biomolecules', halogenated: 'Halogenated',
  inorganic: 'Inorganic', sulfur: 'Sulfur compounds', ions: 'Ions',
  radicals: 'Radicals / open shell', alkenes: 'Alkenes & alkynes',
  large: 'Large molecules', general: 'General',
};

function renderLibrary(mols) {
  const body = $('#libBody');
  const groups = {};
  mols.forEach((m) => {
    const c = m.category || 'general';
    (groups[c] = groups[c] || []).push(m);
  });

  body.innerHTML = '';
  Object.keys(groups).sort().forEach((cat) => {
    const g = el('div', 'lib-group');
    g.appendChild(el('h3', null, CATEGORY_LABEL[cat] || cat));
    groups[cat].forEach((m) => {
      const item = el('div', 'lib-item');
      item.dataset.search = `${m.name} ${m.formula} ${m.key} ${cat}`.toLowerCase();
      item.innerHTML =
        `<div class="top"><span class="nm">${escapeHtml(m.name)}</span>` +
        `<span class="fm">${escapeHtml(m.formula)}</span></div>` +
        `<div class="ds">${escapeHtml(m.desc || '')}</div>`;
      item.addEventListener('click', () => pickLibrary(m));
      g.appendChild(item);
    });
    body.appendChild(g);
  });
}

async function pickLibrary(m) {
  closeLibrary();
  try {
    const mol = await api('/api/molecule/resolve', {
      method: 'POST',
      body: JSON.stringify({ spec: m.key, kind: 'name' }),
    });
    showMolecule(mol);
    renderMoleculeOnly(mol);
    switchTab('geometry');

    // Sync the charge / spin controls to the molecule just loaded, so an ion
    // or radical is not silently computed as a closed-shell singlet.
    $('#inpCharge').value = mol.charge;
    const multSel = $('#inpMult');
    const multOpt = String(mol.multiplicity);
    if ([...multSel.options].some((o) => o.value === multOpt)) {
      multSel.value = multOpt;
    }

    addMessage('assistant',
      `Loaded **${escapeHtml(m.name)}** (${escapeHtml(m.formula)}). ` +
      `Ask me to *optimise* it, get its *HOMO-LUMO gap*, or run *TD-DFT* on it.`);
  } catch (e) {
    addMessage('assistant', `<div class="error-box">${escapeHtml(e.message)}</div>`);
  }
}

function openLibrary()  { $('#libraryDrawer').classList.add('open'); }
function closeLibrary() { $('#libraryDrawer').classList.remove('open'); }

/* ------------------------------------------------------------------ tabs */
function switchTab(name) {
  $$('#resultTabs .tab').forEach((t) =>
    t.classList.toggle('active', t.dataset.tab === name));
  $$('.tab-pane').forEach((p) =>
    p.classList.toggle('active', p.dataset.pane === name));
  // the viewer is shared, so a surface drawn earlier may have been cleared
  // by a later render; redraw on the way back in (grids are cached).
  if (name === 'surfaces' && state.surfaces) applySurface();
  if (name === 'nci' && state.nci) applyNCISurface();
  // a chart drawn while its pane was hidden has no width yet
  if (name === 'series' && state.charts.series) state.charts.series.resize();
  if (name === 'reaction' && state.charts.reaction) state.charts.reaction.resize();
  if (name === 'reaction' && state.charts.irc) state.charts.irc.resize();
  if (name === 'spectrum' && state.charts.uv) state.charts.uv.resize();
  // Same for the three that were left out: their curves are drawn right after
  // the job finishes, while the tab showing them is usually not the active
  // one, so they came up zero-width and stayed that way.
  if (name === 'scan' && state.charts.scan) state.charts.scan.resize();
  if (name === 'vibrations' && state.charts.vib) state.charts.vib.resize();
  if (name === 'vibrations' && state.charts.raman) state.charts.raman.resize();
  if (name === 'nmr' && state.charts.nmr) state.charts.nmr.resize();
  if (name === 'dos' && state.charts.dos) state.charts.dos.resize();
}

/* ------------------------------------------------------------------ composer */
function autoGrow(ta) {
  ta.style.height = 'auto';
  ta.style.height = Math.min(120, ta.scrollHeight) + 'px';
}

/* ------------------------------------------------------------------ wiring */
function wire() {
  // delegated, because the export buttons are static markup repeated once
  // per chart and there is no reason to wire each one by hand
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-export]');
    if (!btn) return;
    const chart = chartById(btn.dataset.chart);
    if (!chart) return;
    if (btn.dataset.export === 'png') exportChartPng(chart, btn.dataset.name);
    else exportChartCsv(chart, btn.dataset.name);
  });

  // Switching the absorption figure between the relative and the absolute
  // ordinate.  Redrawn from the stored result rather than from the chart's
  // own data, so the axis title and the tooltip units move with it.
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-uv-scale]');
    if (!btn) return;
    state.uvScale = btn.dataset.uvScale === 'epsilon' ? 'epsilon' : 'relative';
    $$('#uvScale [data-uv-scale]').forEach((b) => {
      b.classList.toggle('active', b.dataset.uvScale === state.uvScale);
    });
    if (state.lastResult) renderUvCurve(state.lastResult);
  });

  // Switching the NMR figure between isotopes.  Redrawn from the stored
  // payload rather than from the chart, so the axis, the linewidth and the
  // sticks all move together -- the Larmor frequency differs per isotope, so
  // the ppm width of the same 1 Hz line does too.
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-nmr-isotope]');
    if (!btn || !nmrPayload) return;
    nmrIsotope = btn.dataset.nmrIsotope;
    renderNmr(nmrPayload);
  });

  $('#composer').addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage($('#input').value);
  });

  $('#input').addEventListener('input', (e) => autoGrow(e.target));
  $('#input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage($('#input').value);
    }
  });

  $$('#suggestions .chip').forEach((c) => {
    c.addEventListener('click', () => sendMessage(c.dataset.q));
  });

  $('#designCsv')?.addEventListener('click', () => designToCsv());
  $('#seriesCsv')?.addEventListener('click', () => seriesToCsv());
  $('#reactionCsv')?.addEventListener('click', () => reactionToCsv());
  $('#ircCsv')?.addEventListener('click', () => ircToCsv());
  $('#fieldRedraw')?.addEventListener('click', () => fieldRecompute());
  $('#fieldPng')?.addEventListener('click', () => fieldToPng());
  $('#fieldCsv')?.addEventListener('click', () => fieldToCsv());
  $('#fieldContours')?.addEventListener('change', () => drawField());
  $('#seriesProp')?.addEventListener('change', (e) => {
    state.seriesProp = e.target.value;
    drawSeriesChart();
  });

  $('#btnClear').addEventListener('click', () => {
    chatScroll.innerHTML = '';
    addMessage('assistant',
      '<p>History cleared. What would you like to compute?</p>');
  });

  $$('#styleSeg .seg-btn').forEach((b) => {
    b.addEventListener('click', () => {
      $$('#styleSeg .seg-btn').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      state.style = b.dataset.style;
      if (state.viewer) {
        state.viewer.setStyle({}, viewerStyle());
        state.viewer.render();
      }
    });
  });

  $('#btnSpin').addEventListener('click', (e) => {
    state.spin = !state.spin;
    e.target.classList.toggle('primary', state.spin);
    if (state.viewer) {
      if (state.spin) state.viewer.spin('y'); else state.viewer.spin(false);
    }
  });

  $('#btnReset').addEventListener('click', () => {
    if (state.viewer) { state.viewer.zoomTo(); state.viewer.zoom(1.15); state.viewer.render(); }
  });

  $('#btnExport').addEventListener('click', () => {
    if (!state.molecule) return;
    const xyz = state.lastResult && state.lastResult.optimized_xyz
      ? state.lastResult.optimized_xyz
      : molToXyz(state.molecule);
    const blob = new Blob([xyz], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (state.molecule.name || state.molecule.formula || 'molecule') + '.xyz';
    a.click();
    URL.revokeObjectURL(a.href);
  });

  $$('#resultTabs .tab').forEach((t) => {
    t.addEventListener('click', () => switchTab(t.dataset.tab));
  });

  $('#btnLib').addEventListener('click', openLibrary);
  $('#btnCloseLib').addEventListener('click', closeLibrary);

  $('#libSearch').addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase().trim();
    $$('#libBody .lib-item').forEach((it) => {
      it.style.display = !q || it.dataset.search.includes(q) ? '' : 'none';
    });
    $$('#libBody .lib-group').forEach((g) => {
      const any = Array.from(g.querySelectorAll('.lib-item'))
        .some((it) => it.style.display !== 'none');
      g.style.display = any ? '' : 'none';
    });
  });

  $('#btnHelp').addEventListener('click', () => $('#helpModal').classList.add('open'));
  $('#btnCloseHelp').addEventListener('click', () => $('#helpModal').classList.remove('open'));
  $('#helpModal').addEventListener('click', (e) => {
    if (e.target.id === 'helpModal') $('#helpModal').classList.remove('open');
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      $('#helpModal').classList.remove('open');
      closeLibrary();
    }
    if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      $('#input').focus();
    }
  });

  $('#btnClearJobs').addEventListener('click', async () => {
    const finished = state.jobs.filter((j) => j.status === 'completed' || j.status === 'failed');
    for (const j of finished) {
      try { await api(`/api/job/${j.id}`, { method: 'DELETE' }); } catch { /* ignore */ }
    }
    refreshJobs();
  });
}

/* ------------------------------------------------------------------ go */
wire();
boot();
