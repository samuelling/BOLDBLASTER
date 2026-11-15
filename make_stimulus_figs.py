"""One picture per experiment for the dashboard: what each design's stimulus looked like, as frames on
the 24 x 18 deg scanner screen (192 x 144 px). Writes dashboard_data/fig_<experiment>.jpg."""
import os
import numpy as np
from PIL import Image

from boldblaster import game, controlled as ctl
import run_long as RL

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_data')
GAP = 4


def row(frames):
    fs = [np.clip(f, 0, 1) for f in frames]
    out = fs[0]
    for f in fs[1:]:
        out = np.concatenate([out, np.ones((f.shape[0], GAP)), f], 1)
    return out


def stack(rows):
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1])), constant_values=1) for r in rows]
    out = rows[0]
    for r in rows[1:]:
        out = np.concatenate([out, np.ones((GAP, w)), r], 0)
    return out


def save(img, name, scale=1):
    im = Image.fromarray((img * 255).astype(np.uint8))
    if scale != 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    im.save(os.path.join(OUT, f'fig_{name}.jpg'), quality=82, optimize=True)


def frames_of(fn, picks):
    got = []
    sink = lambda ch: got.append(ch)
    fn(sink)
    F = np.concatenate(got)
    return [F[i] for i in picks]


def main():
    # 1 + 2: the raycaster as first built, and with the round-3 design levers
    plain = game.simulate_run_v2(300, duration=40.0, keep_frames=400)['frames']
    lev = game.simulate_run_v2(800, duration=40.0, keep_frames=400, levers=True)['frames']
    save(stack([row([plain[i] for i in (30, 150, 300)]), row([lev[i] for i in (30, 150, 300)])]), 'models')
    # 3: five of the ten level themes
    lv = []
    for th in RL.THEMES[:5]:
        lv.append(game.simulate_run_v2(5000, duration=8.0, keep_frames=80, theme=th)['frames'][60])
    save(row(lv), 'hours')
    # 4: the dedicated V1 battery, one stimulus-on frame each (pSFT, CRF, size, temporal, attention)
    ded = []
    for kind in ('psft', 'crf', 'size', 'temporal', 'attn'):
        got = []
        ctl.run_v1_experiment(kind, 11, duration=40.0, frame_sink=lambda ch: got.append(ch))
        F = np.concatenate(got)
        sd = F.reshape(len(F), -1).std(1)
        if kind == 'size':      # show a disc or annulus: a frame that is partly mean grey
            grey = (np.abs(F - 0.5) < 0.004).reshape(len(F), -1).mean(1)
            ok = np.where((grey > 0.35) & (grey < 0.9) & (sd > 0.05))[0]
            ded.append(F[ok[len(ok) // 3]] if len(ok) else F[0])
        else:
            ded.append(F[int(np.argmax(sd > 0.08 * sd.max()) + 3)] if sd.max() > 0 else F[0])
    save(row(ded), 'dedicated')
    # 6 (round 6): a frame and three of the ventral population's content maps (NPCs, one wall type, floor)
    cats = []
    r = game.simulate_run_v2(301, duration=20.0, keep_frames=200, content_sink=lambda ch: cats.append(ch))
    C = np.concatenate(cats)
    k = int(np.argmax(C[:, 0].reshape(len(C), -1).sum(1)))
    save(row([r['frames'][k], C[k, 0], C[k, 1] * 0.85 + 0.075, C[k, 5] * 0.85 + 0.075]), 'ventral')
    # 7 (round 7): one sequential SEQ-SIM trial, frame by frame
    got = []
    res = ctl.run_seqsim_experiment(3, duration=30.0, frame_sink=lambda ch: got.append(ch))
    F = np.concatenate(got)
    t0, m, s = res['design']['trials'][[i for i, tr in enumerate(res['design']['trials']) if tr[1] == 0 and tr[2] >= 0.3][0]]
    f0 = int(round(t0 * game.FPS))
    save(row([F[f] for f in range(f0 - 1, f0 + 11, 2)]), 'seqsim', scale=0.8)
    # 8 (round 8): the four games, one row each
    rows = []
    for g in ('raycaster', 'betamax', 'brickninja', 'sanctuary'):
        p = os.path.join(OUT, f'round8_frames_{g}.png')
        if os.path.exists(p):
            a = np.asarray(Image.open(p).convert('L'), np.float32) / 255
            rows.append(np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((a.shape[1] // 2, a.shape[0] // 2), Image.NEAREST), np.float32) / 255)
    if rows:
        save(stack(rows), 'games', scale=0.64)
    # round 1: the report's own sandbox figure, top row (game frames with gaze, and a bar frame)
    fig = Image.open(os.path.join(os.path.dirname(OUT), 'docs', 'fig1_sandbox.png')).convert('RGB')
    fig = fig.crop((0, 0, fig.width, int(fig.height * 0.43)))
    fig.resize((fig.width // 2, fig.height // 2), Image.LANCZOS).save(os.path.join(OUT, 'fig_prf.jpg'), quality=80, optimize=True)


if __name__ == '__main__':
    main()
