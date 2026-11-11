// Minimal stand-in for the claude.ai dashboard runtime, so the same page and scripts
// run as a static site (GitHub Pages). Loads the CSVs, runs calcs and drawings, and
// re-draws on control changes and resizes.
(function () {
  const FILES = {
    recovery: 'data/recovery.csv', v1_voxels: 'data/v1_voxels.csv', prf_round1: 'data/prf_round1.csv',
    prf_voxels: 'data/prf_voxels.csv', signal: 'data/signal_components.csv', noise_model: 'data/noise_model.csv',
    diagnostics: 'data/diagnostics.csv', fair: 'data/fair.csv', budget: 'data/budget.csv',
  };
  const DEFAULTS = { param: 'mu', design: 'long_20h', ecc: 'all', r1: 'game_eyetracker' };
  const store = {};
  const calcs = {}, calcCache = {};
  const draws = [];
  const onDataFns = [];
  let params = Object.assign({}, DEFAULTS);
  Object.keys(FILES).forEach((id) => { store[id] = { status: 'loading', data: [] }; });

  function calcValue(id) {
    const c = calcs[id];
    if (c.inputs.some((i) => (store[i] || {}).status !== 'ok')) return { status: 'loading', data: null };
    if (!(id in calcCache)) calcCache[id] = c.fn(...c.inputs.map((i) => store[i].data));
    return { status: 'ok', data: calcCache[id] };
  }

  const cs = getComputedStyle(document.documentElement);
  const colors = [1, 2, 3, 4, 5, 6, 7, 8].map((i) => `var(--bb-series-${i})`);

  window.dash = {
    colors,
    data(id) {
      if (store[id]) return store[id];
      if (calcs[id]) return calcValue(id);
      return { status: 'missing', data: [] };
    },
    calc(id, spec) { calcs[id] = spec; },
    loader() {},
    draw(name, fn) { draws.push([name, fn]); },
    onData(fn) { onDataFns.push(fn); },
    params() { return Object.assign({}, params); },
    setParams(p) { params = Object.assign({}, params, p); schedule(); },
    resetParams() { params = Object.assign({}, DEFAULTS); schedule(); },
    setLink() {},
    refresh() {},
  };

  function runAll() {
    onDataFns.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
    draws.forEach(([name, fn]) => {
      document.querySelectorAll(`[data-draw="${name}"]`).forEach((el) => {
        if (el.closest('[hidden]')) return;
        try { fn(el); } catch (e) { console.error('draw', name, e); }
      });
    });
  }
  let pending = false;
  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; runAll(); });
  }
  window.addEventListener('resize', () => { clearTimeout(window.__bbResize); window.__bbResize = setTimeout(schedule, 120); });
  document.addEventListener('toggle', schedule, true);

  window.__bbStart = function () {
    schedule();
    Object.entries(FILES).forEach(([id, url]) => {
      fetch(url).then((r) => { if (!r.ok) throw new Error(r.status); return r.text(); })
        .then((t) => {
          const rows = d3.csvParse(t, (row) => { for (const k in row) if (row[k] === '') row[k] = null; return row; });
          store[id] = { status: 'ok', data: rows };
          Object.keys(calcCache).forEach((k) => delete calcCache[k]);
          schedule();
        })
        .catch(() => { store[id] = { status: 'error', data: [] }; schedule(); });
    });
  };
})();
