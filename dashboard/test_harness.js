const fs = require('fs'), path = require('path');
const { JSDOM } = require('jsdom');
const dsv = require('d3-dsv');
const D = '/home/claude/BOLDBLASTER/dashboard', DATA = '/home/claude/BOLDBLASTER/dashboard_data';
const html = fs.readFileSync(path.join(D, 'index.html'), 'utf8').replace(/<script src="[^"]+"><\/script>/g, '');
const dom = new JSDOM(`<!DOCTYPE html><body><div id="dash-root">${html}</div></body>`, { runScripts: 'outside-only', pretendToBeVisual: true });
const w = dom.window;
w.eval(fs.readFileSync('/tmp/dashtest/node_modules/d3/dist/d3.min.js', 'utf8'));
// jsdom has no layout: fake widths and text measurement
Object.defineProperty(w.HTMLElement.prototype, 'clientWidth', { get() { return 720; } });
w.SVGElement.prototype.getComputedTextLength = function () { return (this.textContent || '').length * 6.5; };
const files = { recovery: 'recovery.csv', v1_voxels: 'v1_voxels.csv', prf_round1: 'prf_round1.csv', prf_voxels: 'prf_voxels.csv',
  signal: 'signal_components.csv', noise_model: 'noise_model.csv', diagnostics: 'diagnostics.csv' };
const data = {};
for (const [k, f] of Object.entries(files)) { const rows = dsv.csvParse(fs.readFileSync(path.join(DATA, f), 'utf8')); data[k] = rows.map(r => { const o = {}; for (const c of rows.columns) o[c] = r[c] === '' ? null : r[c]; return o; }); }
const calcs = {}, draws = {}, onData = [];
let params = { param: 'mu', design: 'long_20h', ecc: 'all', r1: 'game_eyetracker' };
const errors = [];
w.dash = {
  colors: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
  data(id) { if (data[id]) return { status: 'ok', data: data[id] }; if (calcs[id]) { const c = calcs[id]; return { status: 'ok', data: c.fn(...c.inputs.map(i => data[i])) }; } return { status: 'missing', data: [] }; },
  calc(id, spec) { calcs[id] = spec; }, draw(name, fn) { draws[name] = fn; }, onData(fn) { onData.push(fn); },
  params() { return { ...params }; }, setParams(p) { params = { ...params, ...p }; }, resetParams() {}, setLink() {},
};
w.eval(fs.readFileSync(path.join(D, 'core.js'), 'utf8'));
w.eval(fs.readFileSync(path.join(D, 'charts.js'), 'utf8'));
function runAll(label) {
  onData.forEach(fn => fn());
  for (const [name, fn] of Object.entries(draws)) {
    for (const el of w.document.querySelectorAll(`[data-draw="${name}"]`)) { try { fn(el); } catch (e) { errors.push(`${label} draw ${name}: ${e.message}`); } }
  }
}
runAll('default');
const q = (s) => w.document.querySelector(s);
console.log('takeaway:', q('#takeaway').textContent.replace(/\s+/g, ' ').trim());
console.log('kpis:', [...w.document.querySelectorAll('[data-draw="kpi"]')].map(e => e.textContent).join(' | '));
console.log('scatter-r:', q('[data-draw="scatter-r"]').textContent, q('[data-draw="scatter-r"]').getAttribute('data-where'));
console.log('forest circles:', q('#forest').querySelectorAll('circle').length, ' hours pts:', q('#hours-chart').querySelectorAll('circle').length,
  ' scatter pts:', q('#scatter').querySelectorAll('circle').length, ' prf pts:', q('#prf-chart').querySelectorAll('circle').length);
console.log('prf rows:', q('[data-draw="prftable"]').children.length, ' noise rows:', q('[data-draw="noisetable"]').children.length, ' diag rows:', q('[data-draw="diagtable"]').children.length,
  ' signal bars:', q('#signal-chart').querySelectorAll('rect').length);
for (const p of ['sig', 'c50_eff', 'q', 'k', 'tau', 'a', 'num_mu', 'rpe_rpe_beta']) { params.param = p; runAll(p); }
params.param = 'c50_eff'; params.design = 'dedicated'; params.ecc = '3-7'; runAll('c50 ded ecc');
console.log('c50 dedicated 3-7 pts:', q('#scatter').querySelectorAll('circle').length, 'r:', q('[data-draw="scatter-r"]').textContent);
params.param = 'num_mu'; runAll('nonv1');
console.log('nonv1 notes visible:', [...w.document.querySelectorAll('[data-when="nonv1"]')].map(n => !n.hidden).join(','), ' forest rows:', q('#forest').querySelectorAll('circle').length);
for (const d of ['bars', 'game_et_2h', 'bars+game', 'game_nogaze']) { params.r1 = d; runAll('r1 ' + d); }
console.log('errors:', errors.length ? errors : 'none');
