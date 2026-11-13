// ---------------------------------------------------------------- shared helpers
window.BB = (function () {
  const num = (v) => (v === null || v === undefined || v === '' ? NaN : +v);
  const f2 = d3.format('.2f');
  const V1 = ['mu', 'sig', 'c50_eff', 'q', 'k', 'tau', 'a'];
  const DEFAULTS = { param: 'mu', design: 'long_20h', ecc: 'all', r1: 'game_eyetracker' };
  const params = () => Object.assign({}, DEFAULTS, dash.params());
  // family index into dash.colors: dedicated, game, levers, battery, ten levels
  const FAM_IDX = { dedicated: 0, game: 1, levers: 2, battery: 4, long: 6 };
  function family(design) {
    if (design.startsWith('dedicated')) return 'dedicated';
    if (design.startsWith('levers')) return 'levers';
    if (design.startsWith('battery')) return 'battery';
    if (design.startsWith('long') || design.endsWith('_4h')) return 'long';
    return 'game';
  }
  const color = (fam) => dash.colors[(FAM_IDX[fam] || 0) % dash.colors.length];

  // Status gate for a drawing: returns true when every source is ok.
  function ready(el, ids) {
    for (const id of ids) {
      const s = dash.data(id).status;
      if (s !== 'ok') {
        el.replaceChildren();
        if (s === 'loading') { el.classList.add('dash-skeleton'); }
        else {
          el.classList.remove('dash-skeleton');
          const p = document.createElement('div');
          p.className = 'bb-empty';
          p.textContent = s === 'error' ? 'Failed to load' : '—';
          el.appendChild(p);
        }
        return false;
      }
    }
    el.classList.remove('dash-skeleton');
    return true;
  }

  // Tooltip living inside a chart's drawing element.
  function tip(el) {
    const t = document.createElement('div');
    t.className = 'bb-tip';
    t.hidden = true;
    el.appendChild(t);
    return {
      show(evt, lines) {
        t.replaceChildren(...lines.filter(Boolean).map((l, i) => {
          const d = document.createElement('div');
          d.textContent = l;
          if (i === 0) d.className = 'bb-tip-h';
          return d;
        }));
        t.hidden = false;
        const r = el.getBoundingClientRect();
        let x = evt.clientX - r.left + 14, y = evt.clientY - r.top + 14;
        if (x + t.offsetWidth > r.width) x = evt.clientX - r.left - t.offsetWidth - 14;
        if (y + t.offsetHeight > r.height + 40) y = evt.clientY - r.top - t.offsetHeight - 10;
        t.style.left = Math.max(0, x) + 'px';
        t.style.top = Math.max(0, y) + 'px';
      },
      hide() { t.hidden = true; },
    };
  }

  function svg(el, height, label) {
    const w = Math.max(280, el.clientWidth);
    const s = d3.select(el).append('svg').attr('width', '100%').attr('height', height)
      .attr('viewBox', `0 0 ${w} ${height}`).attr('role', 'img').attr('aria-label', label);
    return { s, w, h: height };
  }

  function styleAxis(g) {
    g.selectAll('path, line').style('stroke', 'var(--cds-chart-axis)');
    g.selectAll('text').style('fill', 'var(--cds-text-secondary)').style('font-size', 'var(--cds-font-size-caption)');
  }

  function grid(g, scale, ticks, x0, x1, horizontal) {
    g.selectAll('line.bb-grid').data(ticks).join('line').attr('class', 'bb-grid')
      .attr(horizontal ? 'y1' : 'x1', (d) => scale(d)).attr(horizontal ? 'y2' : 'x2', (d) => scale(d))
      .attr(horizontal ? 'x1' : 'y1', x0).attr(horizontal ? 'x2' : 'y2', x1)
      .style('stroke', 'var(--cds-chart-grid)');
  }

  function textWidth(svgSel, strings) {
    const t = svgSel.append('text').style('font-size', 'var(--cds-font-size-caption)');
    let m = 0;
    for (const s of strings) { t.text(s); m = Math.max(m, t.node().getComputedTextLength() || s.length * 6.2); }
    t.remove();
    return m;
  }

  const recRow = (rows, design, param) => rows.find((r) => r.design === design && r.parameter === param);
  // data-where "a=1&b=2" (values percent-encoded) -> {a: '1', b: '2'}
  function where(el) {
    const out = {};
    (el.getAttribute('data-where') || '').split('&').filter(Boolean).forEach((kv) => {
      const i = kv.indexOf('=');
      out[decodeURIComponent(kv.slice(0, i))] = decodeURIComponent(kv.slice(i + 1));
    });
    return out;
  }
  const match = (row, w) => Object.keys(w).every((k) => String(row[k]) === w[k]);
  // realistic designs only (oracles use true gaze and pRFs)
  const realistic = (r) => !r.design.endsWith('oracle');

  return { num, f2, V1, params, family, color, ready, tip, svg, styleAxis, grid, textWidth, recRow, where, match, realistic };
})();

// ---------------------------------------------------------------- controls
const bbBound = new WeakSet();
dash.onData(() => {
  const P = BB.params();
  document.querySelectorAll('select[data-param]').forEach((sel) => {
    const name = sel.getAttribute('data-param');
    if (document.activeElement !== sel && P[name] !== undefined) sel.value = P[name];
    if (!bbBound.has(sel)) {
      bbBound.add(sel);
      sel.addEventListener('change', () => dash.setParams({ [name]: sel.value }));
    }
  });
  const nonV1 = !BB.V1.includes(P.param);
  document.querySelectorAll('[data-when="nonv1"]').forEach((n) => { n.hidden = !nonV1; });
  document.querySelectorAll('.bb-legend span[data-fam]').forEach((s) => {
    s.style.setProperty('--bb-c', BB.color(s.getAttribute('data-fam')));
  });
});

// ---------------------------------------------------------------- derived values (shown in the takeaway)
dash.calc('variety', {
  title: 'Largest effect of level variety at equal time',
  description: 'Largest absolute difference in Spearman r between 4 h spread over ten level types and 4 h of one level type, across V1 parameters.',
  inputs: ['recovery'],
  fn: (recovery) => {
    const r = (design, p) => +(recovery.find((x) => x.design === design && x.parameter === p) || {}).spearman;
    return Math.max(...['mu', 'sig', 'c50_eff', 'q', 'k', 'tau', 'a'].map((p) => Math.abs(r('varied_4h', p) - r('single_4h', p))));
  },
});
dash.calc('signal_medians', {
  title: 'Median signal and noise components',
  description: 'Median over round-1 voxels of the SD over time of each component, in percent signal change.',
  inputs: ['signal'],
  fn: (signal) => {
    const med = (col) => d3.median(signal, (r) => +r[col]);
    return [
      { component: 'visual_bars', label: 'Visual, bars', median_sd: med('visual_sd_bars') },
      { component: 'visual_game', label: 'Visual, gameplay', median_sd: med('visual_sd_game') },
      { component: 'systemic', label: 'Systemic arousal, gameplay', median_sd: med('systemic_sd_game') },
      { component: 'noise', label: 'Measurement noise (AR(1) SD)', median_sd: med('noise_sd') },
    ];
  },
});

dash.calc('h2h', {
  title: 'Head-to-head verdicts by budget',
  description: 'For each matched budget (1 h per experiment, 5 h in total), how many of the seven V1 parameters gameplay wins, the dedicated design wins, or tie: a win is a paired 95% interval on the difference in Spearman r that excludes zero.',
  inputs: ['budget'],
  fn: (budget) => {
    const out = [];
    for (const b of ['1 h', '5 h']) for (const v of ['Gameplay', 'Dedicated', 'Tie']) {
      out.push({ budget: b, verdict: v, n: budget.filter((r) => r.budget === b && r.verdict === v).length });
    }
    return out;
  },
});

function drawCalc(id, fmt) {
  dash.draw(id, (el) => {
    const c = dash.data(id);
    el.textContent = c.status === 'ok' && c.data !== null ? fmt(c.data) : '—';
  });
}
drawCalc('variety', (v) => (+v).toFixed(2));

// ---------------------------------------------------------------- tiles
dash.draw('kpi', (el) => {
  const d = dash.data('recovery');
  if (d.status !== 'ok') { el.textContent = '—'; return; }
  const w = BB.where(el);
  const row = BB.recRow(d.data, w.design, w.parameter);
  el.textContent = row ? BB.f2(+row.spearman) : '—';
});

// ---------------------------------------------------------------- any single cell: data-source, data-field, data-where
dash.draw('cell', (el) => {
  const src = (el.getAttribute('data-source') || '').split(' ')[0];
  const d = dash.data(src);
  const rows = d.status === 'ok' && Array.isArray(d.data) ? d.data : [];
  const row = rows.find((r) => BB.match(r, BB.where(el)));
  const v = row ? +row[el.getAttribute('data-field')] : NaN;
  el.textContent = isFinite(v) ? BB.f2(v) : '—';
});

// integer count from a rows calc or dataset: data-source, data-field, data-where
dash.draw('count', (el) => {
  const src = (el.getAttribute('data-source') || '').split(' ')[0];
  const d = dash.data(src);
  const rows = d.status === 'ok' && Array.isArray(d.data) ? d.data : [];
  const row = rows.find((r) => BB.match(r, BB.where(el)));
  el.textContent = row ? String(+row[el.getAttribute('data-field')]) : '—';
});

// ---------------------------------------------------------------- best realistic design for the chosen parameter
function bbBest() {
  const d = dash.data('recovery');
  if (d.status !== 'ok') return null;
  const P = BB.params();
  const rows = d.data.filter((r) => r.parameter === P.param && BB.realistic(r) && !r.design.startsWith('dedicated'));
  const best = rows.reduce((a, r) => (!a || +r.spearman > +a.spearman ? r : a), null);
  const ded = d.data.find((r) => r.parameter === P.param && r.design === 'dedicated');
  const ded60 = d.data.find((r) => r.parameter === P.param && r.design === 'dedicated_60m');
  return { best, ded, ded60 };
}
function bbMark(el, row) {
  if (row) el.setAttribute('data-where', `design=${encodeURIComponent(row.design)}&parameter=${encodeURIComponent(row.parameter)}`);
}
dash.draw('sel-param', (el) => { const b = bbBest(); const r = b && (b.best || b.ded); bbMark(el, r); el.textContent = r ? r.parameter_label : '—'; });
dash.draw('best-design', (el) => { const b = bbBest(); bbMark(el, b && b.best); el.textContent = b && b.best ? b.best.design_label : '—'; });
dash.draw('best-r', (el) => { const b = bbBest(); bbMark(el, b && b.best); el.textContent = b && b.best ? BB.f2(+b.best.spearman) : '—'; });
dash.draw('ded-r', (el) => { const b = bbBest(); bbMark(el, b && b.ded); el.textContent = b && b.ded ? BB.f2(+b.ded.spearman) : '—'; });
dash.draw('ded60-r', (el) => { const b = bbBest(); bbMark(el, b && b.ded60); el.textContent = b && b.ded60 ? BB.f2(+b.ded60.spearman) : '—'; });

// ---------------------------------------------------------------- selected-row numbers
dash.draw('scatter-r', (el) => {
  const P = BB.params();
  const d = dash.data('recovery');
  el.setAttribute('data-where', `design=${encodeURIComponent(P.design)}&parameter=${encodeURIComponent(P.param)}`);
  const row = d.status === 'ok' ? BB.recRow(d.data, P.design, P.param) : null;
  el.textContent = row ? BB.f2(+row.spearman) : '—';
});

dash.draw('prf-label', (el) => {
  const P = BB.params();
  const d = dash.data('prf_round1');
  el.setAttribute('data-where', `design=${encodeURIComponent(P.r1)}`);
  const row = d.status === 'ok' ? d.data.find((r) => r.design === P.r1) : null;
  el.textContent = row ? row.design_label : '—';
});

// ---------------------------------------------------------------- tables
function td(text, right) {
  const c = document.createElement('td');
  c.textContent = text;
  if (right) c.className = 'bb-r';
  return c;
}

dash.draw('prftable', (tb) => {
  const d = dash.data('prf_round1');
  tb.replaceChildren();
  if (d.status !== 'ok') return;
  const P = BB.params();
  const order = ['bars', 'bars_2run', 'bars_1run', 'game_truegaze', 'game_eyetracker', 'game_nogaze', 'game_et_noarousal',
    'bars+game', 'game_et_1run', 'game_et_2run', 'game_et_4run', 'game_et_2h'];
  const rows = d.data.slice().sort((a, b) => order.indexOf(a.design) - order.indexOf(b.design));
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.setAttribute('data-key', encodeURIComponent(r.design));
    tr.className = 'bb-click' + (r.design === P.r1 ? ' bb-sel' : '');
    tr.append(td(r.design_label), td(String(+r.minutes), true), td(BB.f2(+r.pos_err_median), true),
      td(BB.f2(+r.pos_err_ecc_9_12), true), td(BB.f2(+r.r2_frac_ceiling_bar), true));
    tr.addEventListener('click', () => dash.setParams({ r1: r.design }));
    tb.appendChild(tr);
  }
});

dash.draw('budgettable', (tb) => {
  const d = dash.data('budget');
  tb.replaceChildren();
  if (d.status !== 'ok') return;
  const P = BB.params();
  const b = BB.where(tb).budget;
  const fam = { Gameplay: 'long', Dedicated: 'dedicated', Tie: null };
  for (const r of d.data.filter((x) => x.budget === b)) {
    const tr = document.createElement('tr');
    tr.setAttribute('data-key', `${encodeURIComponent(r.budget)},${encodeURIComponent(r.parameter)}`);
    tr.className = 'bb-click' + (r.parameter === P.param ? ' bb-sel' : '');
    const sign = (v) => (v > 0 ? '+' : '') + BB.f2(v);
    const vd = td(r.verdict);
    vd.className = 'bb-verdict';
    vd.style.setProperty('--bb-c', fam[r.verdict] ? BB.color(fam[r.verdict]) : 'var(--cds-chart-muted)');
    tr.append(td(r.parameter_label), td(BB.f2(+r.ded_r), true), td(BB.f2(+r.game_r), true), td(sign(+r.diff), true),
      td(sign(+r.diff_lo), true), td(sign(+r.diff_hi), true), vd);
    tr.addEventListener('click', () => dash.setParams({ param: r.parameter }));
    tb.appendChild(tr);
  }
});

dash.draw('diagtable', (tb) => {
  const d = dash.data('diagnostics');
  tb.replaceChildren();
  if (d.status !== 'ok') return;
  for (const r of d.data) {
    const tr = document.createElement('tr');
    tr.setAttribute('data-key', encodeURIComponent(r.condition));
    tr.append(td(r.condition), td(r.metric), td(BB.f2(+r.value), true));
    tb.appendChild(tr);
  }
});

dash.draw('noisetable', (tb) => {
  const d = dash.data('noise_model');
  tb.replaceChildren();
  if (d.status !== 'ok') return;
  for (const r of d.data) {
    const tr = document.createElement('tr');
    tr.setAttribute('data-key', encodeURIComponent(r.component));
    tr.append(td(r.component), td(r.applies_to), td(r.form), td(r.parameters), td(r.shared_across_voxels), td(r.handled_by_fit));
    tb.appendChild(tr);
  }
});

// ---------------------------------------------------------------- round 6: attention dynamics, ventral gaze tolerance, cue lever
const R6 = (rows) => (part, where) => rows.filter((r) => r.part === part && Object.keys(where || {}).every((k) => String(r[k]) === String(where[k])));

dash.draw('r6attntable', (tb) => {
  const d = dash.data('round6');
  tb.replaceChildren();
  if (d.status !== 'ok') return;
  const by = R6(d.data);
  for (const mins of [20, 40, 60]) {
    const o = by('attn', { condition: 'oracle', minutes: mins });
    const re = by('attn', { condition: 'real', minutes: mins });
    const get = (rows, p) => { const r = rows.find((x) => x.parameter === p); return r ? BB.f2(+r.spearman) : '—'; };
    const tr = document.createElement('tr');
    tr.setAttribute('data-key', String(mins));
    tr.append(td(String(mins), true), td(get(o, 'a'), true), td(get(re, 'a'), true), td(get(o, 'tr'), true),
      td(get(re, 'tr'), true), td(get(o, 'td'), true), td(get(re, 'td'), true));
    tb.appendChild(tr);
  }
});

dash.draw('r6venttable', (tb) => {
  const d = dash.data('round6');
  tb.replaceChildren();
  if (d.status !== 'ok') return;
  const by = R6(d.data);
  for (const mins of [10, 20, 40]) for (const [cond, lab] of [['oracle', 'Oracle gaze+pRF'], ['eyetracker', 'Eye tracker']]) {
    const rows = by('vent', { condition: cond, minutes: mins });
    const get = (p) => { const r = rows.find((x) => x.parameter === p); return r ? BB.f2(+r.spearman) : '—'; };
    const tr = document.createElement('tr');
    tr.setAttribute('data-key', `${mins},${cond}`);
    tr.append(td(String(mins), true), td(lab), td(get('pref_weight'), true), td(get('full_vector_corr'), true), td(get('pref_category_frac'), true));
    tb.appendChild(tr);
  }
});

dash.calc('r6gap', {
  title: 'Gaze-measurement cost at matched 40 minutes, V1 vs the large-RF ventral population',
  description: 'Spearman r under oracle gaze/pRFs minus r under realistic eye-tracker gaze, for the headline parameter of each population, both scored at 40 minutes.',
  inputs: ['round6'],
  fn: (round6) => {
    const get = (part, where) => { const r = round6.find((x) => x.part === part && Object.keys(where).every((k) => String(x[k]) === String(where[k]))); return r ? +r.spearman : NaN; };
    const v1o = get('v1match40', { condition: 'oracle', parameter: 'mu' }), v1r = get('v1match40', { condition: 'realistic', parameter: 'mu' });
    const vo = get('vent', { condition: 'oracle', minutes: 40, parameter: 'pref_weight' }), vr = get('vent', { condition: 'eyetracker', minutes: 40, parameter: 'pref_weight' });
    return [
      { population: 'V1 (pSFT peak)', oracle_r: v1o, realistic_r: v1r, gap: v1o - v1r },
      { population: 'Ventral (preferred category)', oracle_r: vo, realistic_r: vr, gap: vo - vr },
    ];
  },
});
dash.draw('r6gaptable', (tb) => {
  const c = dash.data('r6gap');
  tb.replaceChildren();
  if (c.status !== 'ok' || !c.data) return;
  for (const r of c.data) {
    const tr = document.createElement('tr');
    tr.setAttribute('data-key', encodeURIComponent(r.population));
    tr.append(td(r.population), td(BB.f2(r.oracle_r), true), td(BB.f2(r.realistic_r), true), td(BB.f2(r.gap), true));
    tb.appendChild(tr);
  }
});

dash.draw('r6levertable', (tb) => {
  const d = dash.data('round6');
  tb.replaceChildren();
  if (d.status !== 'ok') return;
  const by = R6(d.data);
  const lev = by('lever', { design: 'lever_cue_1h' }), base = by('lever', { design: 'levers_1h' });
  for (const p of BB.V1) {
    const l = lev.find((x) => x.parameter === p), b = base.find((x) => x.parameter === p);
    if (!l || !b) continue;
    const diff = +l.spearman - +b.spearman;
    const tr = document.createElement('tr');
    tr.setAttribute('data-key', p);
    tr.append(td(l.parameter_label), td(BB.f2(+l.spearman), true), td(BB.f2(+b.spearman), true),
      td((diff > 0 ? '+' : '') + BB.f2(diff), true));
    tb.appendChild(tr);
  }
});
