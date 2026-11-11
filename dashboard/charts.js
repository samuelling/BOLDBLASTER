// ---------------------------------------------------------------- 01 forest plot: recovery by design
dash.draw('forest', (el) => {
  if (!BB.ready(el, ['recovery'])) return;
  el.replaceChildren();
  const P = BB.params();
  const order = ['dedicated', 'dedicated_10min', 'battery_joint', 'game_1h', 'game_2h', 'game_2h_oracle', 'levers_1h',
    'levers_2h', 'levers_2h_oracle', 'long_2h', 'single_4h', 'varied_4h', 'long_5h', 'long_10h', 'long_20h'];
  const rows = dash.data('recovery').data.filter((r) => r.parameter === P.param)
    .sort((a, b) => order.indexOf(a.design) - order.indexOf(b.design));
  if (!rows.length) { const p = document.createElement('div'); p.className = 'bb-empty'; p.textContent = 'No rows for this parameter.'; el.appendChild(p); return; }
  const rowH = 26, top = 26, bottom = 30;
  const H = top + bottom + rows.length * rowH;
  const { s, w } = BB.svg(el, H, `Spearman r by design for the selected parameter, ${rows.length} designs`);
  const labels = rows.map((r) => r.design_label);
  const lw = Math.min(BB.textWidth(s, labels) + 14, w * 0.45);
  const right = 44;
  const x = d3.scaleLinear().domain([Math.min(0, d3.min(rows, (r) => (r.ci_lo !== null && r.ci_lo !== '' ? +r.ci_lo : +r.spearman)) - 0.05), 1]).range([lw, w - right]);
  const y = (i) => top + i * rowH + rowH / 2;
  const g = s.append('g');
  BB.grid(g, x, x.ticks(5), top - 6, H - bottom + 4, false);
  const ax = s.append('g').attr('transform', `translate(0,${H - bottom + 4})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  BB.styleAxis(ax);
  s.append('text').attr('x', w - right).attr('y', top - 10).attr('text-anchor', 'end').text('Spearman r, planted vs fitted');
  rows.forEach((r, i) => {
    if (i > 0 && rows[i - 1].round !== r.round) {
      s.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(i) - rowH / 2).attr('y2', y(i) - rowH / 2)
        .style('stroke', 'var(--color-border-line)');
    }
  });
  // shaded column: the dedicated experiment's 95% interval (round 2-3 voxels)
  const dedRow = rows.find((r) => r.design === 'dedicated');
  if (dedRow && dedRow.ci_lo !== null && dedRow.ci_lo !== '') {
    s.insert('rect', ':first-child').attr('x', x(+dedRow.ci_lo)).attr('width', Math.max(1, x(+dedRow.ci_hi) - x(+dedRow.ci_lo)))
      .attr('y', top - 6).attr('height', H - bottom - top + 10).style('fill', BB.color('dedicated')).style('opacity', 0.1);
  }
  const tip = BB.tip(el);
  const gr = s.append('g').selectAll('g').data(rows).join('g').attr('transform', (r, i) => `translate(0,${y(i)})`);
  gr.append('text').attr('class', 'bb-lab').attr('x', 0).attr('dy', '0.32em').text((r) => r.design_label)
    .each(function () { const t = d3.select(this); let s0 = t.text(); while (this.getComputedTextLength() > lw - 10 && s0.length > 4) { s0 = s0.slice(0, -2); t.text(s0 + '…'); } });
  gr.filter((r) => r.ci_lo !== null && r.ci_lo !== '').append('line')
    .attr('x1', (r) => x(+r.ci_lo)).attr('x2', (r) => x(+r.ci_hi))
    .style('stroke', (r) => BB.color(BB.family(r.design))).style('stroke-width', 2).style('stroke-linecap', 'round');
  gr.append('circle').attr('cx', (r) => x(+r.spearman)).attr('r', 5.5)
    .style('fill', (r) => BB.color(BB.family(r.design))).style('stroke', 'var(--color-panel)').style('stroke-width', 2);
  gr.append('text').attr('x', w).attr('text-anchor', 'end').attr('dy', '0.32em').text((r) => BB.f2(+r.spearman));
  gr.append('rect').attr('x', 0).attr('y', -rowH / 2).attr('width', w).attr('height', rowH).style('fill', 'transparent')
    .on('pointermove', (evt, r) => tip.show(evt, [r.design_label, r.parameter_label, `r = ${BB.f2(+r.spearman)}` +
      (r.ci_lo !== null && r.ci_lo !== '' ? `  [${BB.f2(+r.ci_lo)}, ${BB.f2(+r.ci_hi)}]` : ''), `${+r.minutes} min of data · ${+r.n_voxels} voxels · round ${r.round}`]))
    .on('pointerleave', () => tip.hide());
});

// ---------------------------------------------------------------- 02 hours curve
dash.draw('hours', (el) => {
  const P = BB.params();
  if (!BB.V1.includes(P.param)) { el.replaceChildren(); return; }
  if (!BB.ready(el, ['recovery'])) return;
  el.replaceChildren();
  const rec = dash.data('recovery').data;
  const pts = [['long_2h', 2], ['long_5h', 5], ['long_10h', 10], ['long_20h', 20]]
    .map(([d, h]) => ({ h, row: BB.recRow(rec, d, P.param) })).filter((p) => p.row);
  const ded = BB.recRow(rec, 'dedicated_10min', P.param);
  const H = 300, m = { l: 40, r: 64, t: 18, b: 36 };
  const { s, w } = BB.svg(el, H, 'Spearman r against hours of varied gameplay, with the dedicated experiment as a band');
  const x = d3.scaleLog().domain([1.7, 24]).range([m.l, w - m.r]);
  const y = d3.scaleLinear().domain([-0.1, 1]).range([H - m.b, m.t]);
  BB.grid(s.append('g'), y, y.ticks(5), m.l, w - m.r, true);
  BB.styleAxis(s.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues([2, 5, 10, 20]).tickFormat((d) => d + ' h').tickSizeOuter(0)));
  BB.styleAxis(s.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSizeOuter(0)));
  const cd = BB.color('dedicated'), cl = BB.color('long');
  if (ded) {
    s.append('rect').attr('x', m.l).attr('width', w - m.r - m.l).attr('y', y(+ded.ci_hi)).attr('height', Math.max(1, y(+ded.ci_lo) - y(+ded.ci_hi)))
      .style('fill', cd).style('opacity', 0.12);
    s.append('line').attr('x1', m.l).attr('x2', w - m.r).attr('y1', y(+ded.spearman)).attr('y2', y(+ded.spearman))
      .style('stroke', cd).style('stroke-dasharray', '5 4').style('stroke-width', 1.5);
    s.append('text').attr('x', w - m.r + 6).attr('y', y(+ded.spearman)).attr('dy', '0.32em').text('dedicated');
  }
  const area = d3.area().x((p) => x(p.h)).y0((p) => y(+p.row.ci_lo)).y1((p) => y(+p.row.ci_hi));
  s.append('path').datum(pts).attr('d', area).style('fill', cl).style('opacity', 0.16);
  s.append('path').datum(pts).attr('d', d3.line().x((p) => x(p.h)).y((p) => y(+p.row.spearman)))
    .style('fill', 'none').style('stroke', cl).style('stroke-width', 2.5);
  const tip = BB.tip(el);
  s.append('g').selectAll('circle').data(pts).join('circle').attr('cx', (p) => x(p.h)).attr('cy', (p) => y(+p.row.spearman)).attr('r', 5)
    .style('fill', cl).style('stroke', 'var(--color-panel)').style('stroke-width', 2);
  s.append('g').selectAll('text').data(pts).join('text').attr('x', (p) => x(p.h)).attr('y', (p) => y(+p.row.spearman) - 11)
    .attr('text-anchor', 'middle').text((p) => BB.f2(+p.row.spearman));
  s.append('g').selectAll('rect').data(pts).join('rect').attr('x', (p) => x(p.h) - 18).attr('width', 36).attr('y', m.t).attr('height', H - m.t - m.b)
    .style('fill', 'transparent')
    .on('pointermove', (evt, p) => tip.show(evt, [`${p.h} h of varied gameplay`, `r = ${BB.f2(+p.row.spearman)}  [${BB.f2(+p.row.ci_lo)}, ${BB.f2(+p.row.ci_hi)}]`,
      ded ? `Dedicated 10 min: ${BB.f2(+ded.spearman)}  [${BB.f2(+ded.ci_lo)}, ${BB.f2(+ded.ci_hi)}]` : '']))
    .on('pointerleave', () => tip.hide());
});

// ---------------------------------------------------------------- 02b variety at equal time
dash.draw('varietychart', (el) => {
  if (!BB.ready(el, ['recovery'])) return;
  el.replaceChildren();
  const rec = dash.data('recovery').data;
  const labs = { mu: 'pSFT peak', sig: 'pSFT bandwidth', c50_eff: 'pCRF eff. c50', q: 'pCRF exponent', k: 'Surround k', tau: 'Transient', a: 'Attention gain' };
  const rows = BB.V1.map((p) => ({ p, one: BB.recRow(rec, 'single_4h', p), ten: BB.recRow(rec, 'varied_4h', p) })).filter((r) => r.one && r.ten);
  const P = BB.params();
  const H = 300, rowH = (H - 58) / rows.length;
  const { s, w } = BB.svg(el, H, 'Spearman r at 4 h for one level type versus ten level types');
  const lw = BB.textWidth(s, Object.values(labs)) + 12;
  const x = d3.scaleLinear().domain([0, 1]).range([lw, w - 10]);
  BB.grid(s.append('g'), x, x.ticks(5), 4, H - 44, false);
  BB.styleAxis(s.append('g').attr('transform', `translate(0,${H - 44})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0)));
  const c1 = 'var(--cds-chart-muted)', c2 = BB.color('long');
  const tip = BB.tip(el);
  const g = s.append('g').selectAll('g').data(rows).join('g').attr('transform', (r, i) => `translate(0,${8 + i * rowH + rowH / 2})`);
  g.append('text').attr('class', 'bb-lab').attr('dy', '0.32em').style('font-weight', (r) => (r.p === P.param ? 600 : 400)).text((r) => labs[r.p]);
  g.append('line').attr('x1', (r) => x(Math.min(+r.one.spearman, +r.ten.spearman))).attr('x2', (r) => x(Math.max(+r.one.spearman, +r.ten.spearman)))
    .style('stroke', 'var(--cds-chart-grid)').style('stroke-width', 4).style('stroke-linecap', 'round');
  g.append('circle').attr('cx', (r) => x(+r.one.spearman)).attr('r', 5.5).style('fill', c1).style('stroke', 'var(--color-panel)').style('stroke-width', 2);
  g.append('circle').attr('cx', (r) => x(+r.ten.spearman)).attr('r', 5.5).style('fill', c2).style('stroke', 'var(--color-panel)').style('stroke-width', 2);
  g.append('rect').attr('x', 0).attr('y', -rowH / 2).attr('width', w).attr('height', rowH).style('fill', 'transparent')
    .on('pointermove', (evt, r) => tip.show(evt, [labs[r.p], `One level type: ${BB.f2(+r.one.spearman)}  [${BB.f2(+r.one.ci_lo)}, ${BB.f2(+r.one.ci_hi)}]`,
      `Ten level types: ${BB.f2(+r.ten.spearman)}  [${BB.f2(+r.ten.ci_lo)}, ${BB.f2(+r.ten.ci_hi)}]`]))
    .on('pointerleave', () => tip.hide());
  const lg = s.append('g').attr('transform', `translate(${lw},${H - 6})`);
  [['One level type', c1], ['Ten level types', c2]].forEach(([t, c], i) => {
    const gg = lg.append('g').attr('transform', `translate(${i * 130},0)`);
    gg.append('circle').attr('r', 4.5).attr('cy', -4).style('fill', c);
    gg.append('text').attr('x', 9).text(t);
  });
});

// ---------------------------------------------------------------- 03 planted vs fitted
dash.draw('scatter', (el) => {
  const P = BB.params();
  if (!BB.V1.includes(P.param)) { el.replaceChildren(); return; }
  if (!BB.ready(el, ['v1_voxels'])) return;
  el.replaceChildren();
  const [lo, hi] = P.ecc === 'all' ? [0, 99] : P.ecc.split('-').map(Number);
  const pts = dash.data('v1_voxels').data.filter((r) => r.design === P.design && r.parameter === P.param && +r.ecc >= lo && +r.ecc < hi);
  if (!pts.length) { const p = document.createElement('div'); p.className = 'bb-empty'; p.textContent = 'No voxels for this selection.'; el.appendChild(p); return; }
  const logScale = P.param === 'c50_eff';
  const H = Math.min(440, Math.max(320, el.clientWidth * 0.6)), m = { l: 52, r: 16, t: 14, b: 42 };
  const { s, w } = BB.svg(el, H, `Planted against fitted values for ${pts.length} voxels`);
  const all = pts.flatMap((r) => [+r.planted, +r.fitted]).filter((v) => isFinite(v) && (!logScale || v > 0));
  let dom = d3.extent(all);
  const pad = (dom[1] - dom[0]) * 0.05;
  dom = logScale ? [Math.max(dom[0] * 0.8, 1e-3), dom[1] * 1.25] : [dom[0] - pad, dom[1] + pad];
  const mk = logScale ? d3.scaleLog : d3.scaleLinear;
  const x = mk().domain(dom).range([m.l, w - m.r]).clamp(true);
  const y = mk().domain(dom).range([H - m.b, m.t]).clamp(true);
  BB.grid(s.append('g'), y, y.ticks(5), m.l, w - m.r, true);
  BB.styleAxis(s.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6, logScale ? '~g' : undefined).tickSizeOuter(0)));
  BB.styleAxis(s.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5, logScale ? '~g' : undefined).tickSizeOuter(0)));
  s.append('text').attr('x', (m.l + w - m.r) / 2).attr('y', H - 6).attr('text-anchor', 'middle').text('planted');
  s.append('text').attr('transform', `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('fitted');
  s.append('line').attr('x1', x(dom[0])).attr('y1', y(dom[0])).attr('x2', x(dom[1])).attr('y2', y(dom[1]))
    .style('stroke', 'var(--cds-chart-reference)').style('stroke-dasharray', '5 4');
  const ce = d3.scaleSequential(d3.interpolateViridis).domain([0, 12]);
  const tip = BB.tip(el);
  s.append('g').selectAll('circle').data(pts).join('circle')
    .attr('cx', (r) => x(+r.planted)).attr('cy', (r) => y(+r.fitted)).attr('r', 4)
    .style('fill', (r) => ce(+r.ecc)).style('opacity', 0.8).style('stroke', 'var(--color-panel)').style('stroke-width', 0.8)
    .on('pointermove', (evt, r) => tip.show(evt, [`Voxel ${r.voxel}`, `Eccentricity ${(+r.ecc).toFixed(1)}°, polar angle ${(+r.polar_angle).toFixed(0)}°`,
      `Planted ${(+r.planted).toPrecision(3)}`, `Fitted ${(+r.fitted).toPrecision(3)}`]))
    .on('pointerleave', () => tip.hide());
  const kw = Math.min(160, w * 0.3), kx = w - m.r - kw, ky = m.t + 4;
  const defs = s.append('defs');
  const gid = 'bbecc' + Math.random().toString(36).slice(2, 7);
  const lin = defs.append('linearGradient').attr('id', gid);
  d3.range(0, 1.01, 0.1).forEach((t) => lin.append('stop').attr('offset', t).attr('stop-color', ce(t * 12)));
  s.append('rect').attr('x', kx).attr('y', ky).attr('width', kw).attr('height', 7).attr('rx', 3).style('fill', `url(#${gid})`);
  s.append('text').attr('x', kx).attr('y', ky + 20).text('0°');
  s.append('text').attr('x', kx + kw).attr('y', ky + 20).attr('text-anchor', 'end').text('12° eccentricity');
});

// ---------------------------------------------------------------- 04 round-1 pRF error vs eccentricity
dash.draw('prfscatter', (el) => {
  if (!BB.ready(el, ['prf_voxels', 'prf_round1'])) return;
  el.replaceChildren();
  const P = BB.params();
  const pts = dash.data('prf_voxels').data.filter((r) => r.design === P.r1);
  const sum = dash.data('prf_round1').data.find((r) => r.design === P.r1);
  if (!pts.length) { const p = document.createElement('div'); p.className = 'bb-empty'; p.textContent = 'No voxels for this design.'; el.appendChild(p); return; }
  const H = 320, m = { l: 44, r: 14, t: 12, b: 40 };
  const { s, w } = BB.svg(el, H, 'pRF position error against eccentricity for the selected design');
  const x = d3.scaleLinear().domain([0, 12.5]).range([m.l, w - m.r]);
  const y = d3.scaleLog().domain([0.03, 30]).range([H - m.b, m.t]).clamp(true);
  BB.grid(s.append('g'), y, [0.1, 1, 10], m.l, w - m.r, true);
  BB.styleAxis(s.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat((d) => d + '°').tickSizeOuter(0)));
  BB.styleAxis(s.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).tickValues([0.1, 1, 10]).tickFormat(d3.format('~g')).tickSizeOuter(0)));
  s.append('text').attr('x', (m.l + w - m.r) / 2).attr('y', H - 6).attr('text-anchor', 'middle').text('true eccentricity');
  s.append('text').attr('transform', `translate(11,${(m.t + H - m.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('position error (°)');
  s.append('line').attr('x1', x(9)).attr('x2', x(9)).attr('y1', m.t).attr('y2', H - m.b).style('stroke', 'var(--cds-chart-reference)').style('stroke-dasharray', '4 4');
  s.append('text').attr('x', x(9) + 4).attr('y', m.t + 10).text('bar aperture');
  const c = BB.color(P.r1.startsWith('bars') ? 'dedicated' : 'game');
  const tip = BB.tip(el);
  s.append('g').selectAll('circle').data(pts).join('circle').attr('cx', (r) => x(+r.ecc)).attr('cy', (r) => y(Math.max(+r.pos_err, 0.031)))
    .attr('r', 3).style('fill', c).style('opacity', 0.5)
    .on('pointermove', (evt, r) => tip.show(evt, [`Voxel ${r.voxel}`, `Eccentricity ${(+r.ecc).toFixed(1)}°`, `True centre (${(+r.true_x).toFixed(1)}, ${(+r.true_y).toFixed(1)})°`,
      `Fitted centre (${(+r.fit_x).toFixed(1)}, ${(+r.fit_y).toFixed(1)})°`, `Error ${(+r.pos_err).toFixed(2)}°`]))
    .on('pointerleave', () => tip.hide());
  if (sum) {
    const steps = [[0, 2, +sum.pos_err_ecc_0_2], [2, 5, +sum.pos_err_ecc_2_5], [5, 9, +sum.pos_err_ecc_5_9], [9, 12.1, +sum.pos_err_ecc_9_12]];
    s.append('g').selectAll('line').data(steps).join('line').attr('x1', (d) => x(d[0])).attr('x2', (d) => x(d[1]))
      .attr('y1', (d) => y(d[2])).attr('y2', (d) => y(d[2])).style('stroke', 'var(--color-fg)').style('stroke-width', 2.5);
  }
});

// ---------------------------------------------------------------- 05 signal vs noise
dash.draw('signal', (el) => {
  if (!BB.ready(el, ['signal_medians'])) return;
  el.replaceChildren();
  const rows = dash.data('signal_medians').data || [];
  if (!rows.length) return;
  const H = 240, rowH = (H - 30) / rows.length;
  const { s, w } = BB.svg(el, H, 'Median SD over time of visual signal, systemic signal and noise');
  const lw = BB.textWidth(s, rows.map((r) => r.label)) + 12;
  const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.median_sd) * 1.15]).range([lw, w - 40]);
  BB.grid(s.append('g'), x, x.ticks(4), 0, H - 26, false);
  BB.styleAxis(s.append('g').attr('transform', `translate(0,${H - 26})`).call(d3.axisBottom(x).ticks(4).tickFormat((d) => d + '%').tickSizeOuter(0)));
  const cols = { visual_bars: BB.color('dedicated'), visual_game: BB.color('game'), systemic: 'var(--cds-chart-muted)', noise: 'var(--cds-chart-muted)' };
  const g = s.append('g').selectAll('g').data(rows).join('g').attr('transform', (r, i) => `translate(0,${i * rowH + rowH / 2})`);
  g.append('text').attr('class', 'bb-lab').attr('dy', '0.32em').text((r) => r.label);
  g.append('rect').attr('x', lw).attr('y', -rowH * 0.3).attr('height', rowH * 0.6).attr('rx', 4)
    .attr('width', (r) => Math.max(0, x(r.median_sd) - lw)).style('fill', (r) => cols[r.component])
    .style('opacity', (r) => (r.component === 'noise' ? 0.55 : 1));
  g.append('text').attr('x', (r) => x(r.median_sd) + 6).attr('dy', '0.32em').text((r) => r.median_sd.toFixed(2) + '%');
});
