"""Build the static GitHub Pages version of the dashboard into docs/dashboard/."""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)                       # dashboard/
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SRC, '..', 'docs', 'dashboard')
DATA = os.path.join(SRC, '..', 'dashboard_data')
os.makedirs(os.path.join(OUT, 'data'), exist_ok=True)

page = open(os.path.join(SRC, 'index.html')).read()
page = page.replace('<meta name="dash-one-writer">\n', '')
page = page.replace('<script src="core.js"></script>\n<script src="charts.js"></script>\n', '')

TOKENS = """
:root {
  --font-anthropic-sans: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-anthropic-serif: "Source Serif 4", Georgia, serif;
  --color-fg: #18181a; --color-fg-muted: #66656f; --color-bg: #f7f7f5; --color-panel: #ffffff;
  --color-border-line: #e4e3df; --color-ok: #1baf7a; --color-warn: #eda100; --color-bad: #e34948;
  --cds-font-size-caption: 12.5px; --cds-font-size-body: 15px; --cds-font-size-heading: 18px; --cds-font-size-title: 26px;
  --cds-font-weight-medium: 600;
  --cds-gap-xs: 4px; --cds-gap-sm: 8px; --cds-gap-md: 14px; --cds-gap-lg: 20px; --cds-gap-xl: 28px;
  --cds-pad-xs: 4px; --cds-pad-sm: 8px; --cds-pad-md: 12px; --cds-pad-lg: 18px; --cds-pad-xl: 32px;
  --cds-radius: 12px;
  --cds-chart-grid: #ecebe7; --cds-chart-axis: #c9c7c0; --cds-chart-reference: #9c9a92; --cds-chart-muted: #b4b2aa;
  --cds-text-secondary: #6b6a72; --cds-surface-popover: #ffffff; --cds-border: #e4e3df;
  --cds-shadow-popover: 0 8px 28px rgba(20, 20, 30, .14);
  --bb-series-1: #2a78d6; --bb-series-2: #eb6834; --bb-series-3: #1baf7a; --bb-series-4: #eda100;
  --bb-series-5: #e87ba4; --bb-series-6: #008300; --bb-series-7: #6250d6; --bb-series-8: #e34948;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --color-fg: #f0efec; --color-fg-muted: #a9a8b0; --color-bg: #121214; --color-panel: #1b1b1e;
    --color-border-line: #2e2e33; --cds-chart-grid: #2a2a2f; --cds-chart-axis: #45454c; --cds-chart-reference: #77767e;
    --cds-chart-muted: #6c6b73; --cds-text-secondary: #a3a2aa; --cds-surface-popover: #222226; --cds-border: #34343a;
    --cds-shadow-popover: 0 8px 28px rgba(0, 0, 0, .5);
    --bb-series-1: #3987e5; --bb-series-2: #d95926; --bb-series-3: #199e70; --bb-series-4: #c98500;
    --bb-series-5: #d55181; --bb-series-6: #2a9a2a; --bb-series-7: #9085e9; --bb-series-8: #e66767;
    color-scheme: dark;
  }
}
html, body { margin: 0; background: var(--color-bg); color: var(--color-fg); font-family: var(--font-anthropic-sans); font-size: var(--cds-font-size-body); -webkit-font-smoothing: antialiased; }
#dash-root { padding: 0 clamp(14px, 4vw, 40px) 60px; }
.dash-select { font: inherit; font-size: 14px; color: var(--color-fg); background-color: var(--color-panel); border: 1px solid var(--color-border-line); border-radius: 8px; padding: 6px 30px 6px 10px; appearance: none; -webkit-appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%2388878f' stroke-width='1.5'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: right 10px center; min-width: 210px; }
.dash-skeleton { position: relative; color: transparent !important; }
.dash-skeleton::after { content: ""; position: absolute; inset: 0; border-radius: 8px; background: linear-gradient(90deg, var(--color-border-line), transparent, var(--color-border-line)); background-size: 200% 100%; opacity: .5; animation: bbsk 1.4s linear infinite; }
@keyframes bbsk { to { background-position: -200% 0; } }
.bb-ghbar { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; max-width: 1180px; margin: 0 auto; padding: 14px 0 0; font-size: 12.5px; color: var(--color-fg-muted); }
.bb-ghbar a { color: var(--color-fg); text-decoration: none; border-bottom: 1px solid var(--color-border-line); }
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BOLDBLASTER: what can gameplay fMRI recover?</title>
<meta name="description" content="Simulation testbed for an open fMRI dataset of people playing a first-person game: which tuning models can be recovered, and how much play it takes.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E%3Ccircle cx=%2716%27 cy=%2716%27 r=%2714%27 fill=%27%236250d6%27/%3E%3Ccircle cx=%2716%27 cy=%2716%27 r=%276%27 fill=%27%23eb6834%27/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Source+Serif+4:wght@400&display=swap" rel="stylesheet">
<style>{TOKENS}</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
</head>
<body>
<div class="bb-ghbar"><span>BOLDBLASTER · interactive results</span><span><a href="https://github.com/samuelling/boldblaster">Code &amp; data on GitHub</a> · <a href="../">Full report</a></span></div>
<div id="dash-root">
{page}
</div>
<script src="dash-shim.js"></script>
<script src="core.js"></script>
<script src="charts.js"></script>
<script>window.__bbStart();</script>
</body>
</html>
"""
open(os.path.join(OUT, 'index.html'), 'w').write(html)
for f in ('core.js', 'charts.js'):
    shutil.copy(os.path.join(SRC, f), os.path.join(OUT, f))
shutil.copy(os.path.join(HERE, 'dash-shim.js'), os.path.join(OUT, 'dash-shim.js'))
for f in os.listdir(DATA):
    if f.endswith('.csv'):
        shutil.copy(os.path.join(DATA, f), os.path.join(OUT, 'data', f))
print('built', OUT)
