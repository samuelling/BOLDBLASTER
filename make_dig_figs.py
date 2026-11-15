"""Pictures for the round-10 tab: what the tunnels look like to the models, and the plan of the warren
with one run's path through it."""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = '/home/claude/bbx/runs_dig'
run = int(sys.argv[1]) if len(sys.argv) > 1 else 0
R = [json.loads(l) for l in open(f'{SRC}/betamaxt{run}_log.jsonl')]
ev = json.load(open(f'{SRC}/betamaxt{run}_events.json'))
L = np.memmap(f'{SRC}/betamaxt{run}_lum.u16', dtype=np.uint16, mode='r').reshape(-1, 144, 192)


def pick(cond, skip=0):
    k = [i for i, r in enumerate(R) if cond(r)]
    return k[min(len(k) - 1, len(k) // 2 + skip)] if k else None


fr = [('On the dune, heading for a hatch', pick(lambda r: r['phase'] == 'toHatch')),
      ('In the warren, lights on', pick(lambda r: r['under'] and not r['dark'] and not r['climb'])),
      ('A figure in the gloom', pick(lambda r: r['under'] and any(v[4] >= 1000 and v[3] for v in r['vis']))),
      ('Lights out: lantern only', pick(lambda r: r['under'] and r['dark']))]
fr = [(t, k) for t, k in fr if k is not None]
fig, ax = plt.subplots(1, len(fr), figsize=(2.6 * len(fr), 2.25), dpi=110)
for a, (t, k) in zip(np.atleast_1d(ax), fr):
    a.imshow(L[k].astype(float) / 65535, cmap='gray', vmin=0, vmax=1)
    a.set_title(t, fontsize=8); a.axis('off')
plt.tight_layout(pad=0.3)
plt.savefig('dashboard_data/fig_dig_frames.jpg', pil_kwargs=dict(quality=78))

# the plan: carved corridors, hatches, magazines, and this run's path below ground
P = ev['plan']
SX, SZ, AX0, AZ0, s = P['SX'], P['SZ'], P['AX0'], P['AZ0'], P['DG']['sub']
C = np.array(P['carved'], float).reshape(SZ, SX)
fig, a = plt.subplots(figsize=(7.2, 4.0), dpi=110)
a.imshow(C, cmap='Greys', origin='lower', extent=[AX0, AX0 + SX * s, AZ0, AZ0 + SZ * s], alpha=0.55, vmin=0, vmax=1.6)
import os
stale = []
for k in range(12):
    fl = f'{SRC}/betamaxt{k}_log.jsonl'
    if not os.path.exists(fl):
        continue
    Rk = [json.loads(l) for l in open(fl)]
    stale += json.load(open(f'{SRC}/betamaxt{k}_events.json'))['stale']
    xy = np.array([[r['x'], r['z']] for r in Rk if r['under']])
    if len(xy) > 1:
        seg = np.hypot(*np.diff(xy, axis=0).T) > 3
        xs, zs = xy[:, 0].copy(), xy[:, 1].copy()
        xs[1:][seg] = np.nan
        a.plot(xs, zs, color='#2a78d6', lw=0.8, alpha=0.35)
for h in P['hatches']:
    a.plot(h['x'], h['z'], marker='s', ms=6, color={'lot': '#555', 'line': '#c0392b', 'back': '#8e44ad'}[h['kind']])
for c in P['crates']:
    a.plot(c['x'], c['z'], marker='D', ms=6, color='#d4a017')
for e in stale:
    DG = P['DG']
    (ax_, az_), (bx_, bz_) = [(DG['x0'] + (k % DG['nx']) * DG['cell'], DG['z0'] + (k // DG['nx']) * DG['cell']) for k in (e[1], e[2])]
    a.plot((ax_ + bx_) / 2, (az_ + bz_) / 2, marker='x', ms=8, mew=2, color='#e67e22')
a.set_xlabel('x (m, east is left)', fontsize=8); a.set_ylabel('z (m, the sea is up)', fontsize=8); a.tick_params(labelsize=7)
a.set_title('The Diggings. Grey: tunnels. Squares: hatches (red on the crest). Diamonds: magazines.\nOrange x: cave-ins the player ran into. Blue: an hour of play below ground (12 runs)', fontsize=8)
plt.tight_layout(pad=0.6)
plt.savefig('dashboard_data/fig_dig_plan.jpg', pil_kwargs=dict(quality=80))
print('frames', fr)
