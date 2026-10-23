# HTML tool for picking the winter windows
# save the downloaded winter_windows.json in data/derived/velocity/

import os
import sys
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

PREP_DIR = os.path.join(config.DATA_DIR, 'velocity')
OUTPUT_HTML = os.path.join(PREP_DIR, 'winter_window_selector.html')

# html template
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Winter Window Selector</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-annotation/3.0.1/chartjs-plugin-annotation.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0f1117;--surface:#181d27;--border:#2a3245;--text:#e2e8f2;--muted:#6b7a99;
      --accent:#4a9eff;--win-bg:rgba(74,158,255,.14);--win-bd:rgba(74,158,255,.7);
      --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-weight:300;
     min-height:100vh;display:flex;flex-direction:column}
header{padding:14px 24px 10px;border-bottom:1px solid var(--border);
       display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
header h1{font-family:var(--mono);font-size:13px;font-weight:600;letter-spacing:.12em;
          text-transform:uppercase;color:var(--accent)}
header p{font-size:11px;color:var(--muted);flex:1}
.toolbar{padding:8px 24px;display:flex;align-items:center;gap:8px;
         border-bottom:1px solid var(--border);flex-wrap:wrap}
.roi-nav{display:flex;align-items:center;gap:5px}
button{font-family:var(--mono);font-size:11px;font-weight:600;letter-spacing:.07em;
       text-transform:uppercase;padding:5px 12px;border:1px solid var(--border);
       background:var(--surface);color:var(--text);cursor:pointer;transition:all .15s;border-radius:3px}
button:hover{border-color:var(--accent);color:var(--accent)}
button.primary{background:var(--accent);border-color:var(--accent);color:#0f1117}
button.primary:hover{opacity:.85}
button.danger{border-color:#ef4444;color:#ef4444}
button.danger:hover{background:#ef4444;color:#0f1117}
#roi-label{font-family:var(--mono);font-size:17px;font-weight:600;min-width:60px;text-align:center}
.spacer{flex:1}
#status{font-family:var(--mono);font-size:10px;color:var(--muted);padding:3px 8px;
        background:var(--surface);border:1px solid var(--border);border-radius:3px}
.chart-wrap{flex:1;min-height:360px;padding:8px 24px 6px}
#chartCanvas{cursor:crosshair;width:100%!important;height:100%!important}
.windows-panel{padding:8px 24px 10px;border-top:1px solid var(--border);
               display:flex;gap:8px;flex-wrap:wrap;align-items:center;min-height:44px}
.win-tag{font-family:var(--mono);font-size:10px;padding:3px 9px;
         background:var(--win-bg);border:1px solid var(--win-bd);border-radius:3px;
         color:var(--accent);cursor:pointer;display:flex;align-items:center;gap:5px;transition:all .15s}
.win-tag:hover{background:rgba(74,158,255,.32)}
.win-tag .del{color:var(--muted);font-size:12px;line-height:1}
.win-tag:hover .del{color:#ef4444}
.no-windows{color:var(--muted);font-size:11px;font-family:var(--mono)}
footer{padding:10px 24px;border-top:1px solid var(--border);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
#json-out{font-family:var(--mono);font-size:9.5px;background:var(--surface);
          border:1px solid var(--border);color:var(--text);padding:7px 10px;flex:1;
          max-height:68px;resize:vertical;border-radius:3px;overflow-y:auto;white-space:pre}
.legend{display:flex;gap:14px;padding:2px 24px 4px;flex-wrap:wrap}
.leg-item{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--muted)}
.leg-line{width:20px;height:2px}
.leg-dot{width:7px;height:7px;border-radius:50%}
</style>
</head>
<body>
<header>
  <h1>Winter Window Selector - Ice Velocity</h1>
  <p>Click-drag to mark a winter window &nbsp;·&nbsp; Click a tag to delete it
  &nbsp;·&nbsp; ◀ ▶ or arrow keys to navigate ROIs &nbsp;·&nbsp;
  Baseline / W1–W4 classification is automatic in the plot script</p>
</header>
<div class="toolbar">
  <div class="roi-nav">
    <button onclick="navigate(-1)">◀</button>
    <span id="roi-label"></span>
    <button onclick="navigate(1)">▶</button>
  </div>
  <span id="roi-info" style="font-family:var(--mono);font-size:10px;color:var(--muted)"></span>
  <div class="spacer"></div>
  <button class="danger" onclick="clearCurrentROI()">Clear ROI</button>
  <span id="status">ready</span>
</div>
<div class="legend">
  <div class="leg-item"><div class="leg-line" style="background:#4a9eff"></div>Smoothed</div>
  <div class="leg-item"><div class="leg-dot" style="background:#6b7a99"></div>Raw</div>
  <div class="leg-item">
    <div class="leg-line" style="background:rgba(255,255,255,.35);border-top:1.5px dashed rgba(255,255,255,.4);width:20px"></div>
    Aug 2021 event
  </div>
  <div class="leg-item">
    <div style="width:18px;height:12px;background:var(--win-bg);border:1px solid var(--win-bd);border-radius:2px"></div>
    Selected window
  </div>
</div>
<div class="chart-wrap"><canvas id="chartCanvas"></canvas></div>
<div class="windows-panel" id="windows-panel">
  <span class="no-windows">No windows selected for this ROI</span>
</div>
<footer>
  <button class="primary" onclick="copyJSON()">Copy JSON</button>
  <button onclick="downloadJSON()">Download JSON</button>
  <textarea id="json-out" readonly placeholder="JSON will appear here..."></textarea>
</footer>

<script>
const ALL_DATA   = __DATA_PLACEHOLDER__;
const EVENT_DATE = '2021-08-14T00:00:00';
const X_MIN      = new Date('2016-01-01').getTime();
const X_MAX      = new Date('2026-01-01').getTime();
// Explicit order from Python - guaranteed correct regardless of JS key sorting
const roi_labels = __ORDER_PLACEHOLDER__;

let currentIdx = 0;
const windows  = {};
roi_labels.forEach(l => windows[l] = []);

// Restore from localStorage
try {
  const saved = localStorage.getItem('vel_win_v3');
  if (saved) {
    const parsed = JSON.parse(saved);
    Object.keys(parsed).forEach(k => {
      const lbl = parseInt(k);
      if (windows[lbl] !== undefined)
        windows[lbl] = parsed[k].map(w => [new Date(w[0]).getTime(), new Date(w[1]).getTime()]);
    });
  }
} catch(e) {}

let chart = null;
const tsMs = s => new Date(s).getTime();

function buildAnnotations(lbl) {
  const anns = {};
  anns.event = {
    type: 'line', xMin: tsMs(EVENT_DATE), xMax: tsMs(EVENT_DATE),
    borderColor: 'rgba(255,255,255,0.4)', borderWidth: 1.5, borderDash: [5, 4],
  };
  (windows[lbl] || []).forEach((w, i) => {
    anns[`w${i}`] = {
      type: 'box', xMin: w[0], xMax: w[1],
      backgroundColor: 'rgba(74,158,255,0.13)',
      borderColor: 'rgba(74,158,255,0.65)', borderWidth: 1.2,
    };
  });
  return anns;
}

function renderChart(lbl) {
  const d      = ALL_DATA[lbl];
  const times  = d.times.map(tsMs);
  const raw    = times.map((t, i) => ({ x: t, y: d.v_raw[i] }));
  const smooth = times.map((t, i) => ({ x: t, y: d.v_smooth[i] }));

  const cfg = {
    type: 'line',
    data: { datasets: [
      { label:'Smoothed', data:smooth, borderColor:'#4a9eff', borderWidth:2.2,
        pointRadius:0, tension:0.25, order:1 },
      { label:'Raw', data:raw, borderColor:'transparent', backgroundColor:'#8899bb88',
        pointRadius:2.2, pointStyle:'circle', showLine:false, order:2 },
    ]},
    options: {
      responsive:true, maintainAspectRatio:false,
      animation:{ duration:100 }, parsing:false,
      scales: {
        x: {
          type:'time', min:X_MIN, max:X_MAX,
          time:{ unit:'year', displayFormats:{ year:'yyyy' } },
          grid:{ color:'#1e2535' },
          ticks:{ color:'#6b7a99', font:{ family:'IBM Plex Mono', size:10 }, maxTicksLimit:11 },
        },
        y: {
          grid:{ color:'#1e2535' },
          ticks:{ color:'#6b7a99', font:{ family:'IBM Plex Mono', size:10 } },
          title:{ display:true, text:'Ice speed (m yr\u207b\u00b9)', color:'#6b7a99',
                  font:{ family:'IBM Plex Sans', size:11 } },
        },
      },
      plugins: {
        legend:{ display:false },
        tooltip:{
          mode:'nearest', intersect:false,
          backgroundColor:'#181d27', borderColor:'#2a3245', borderWidth:1,
          titleColor:'#e2e8f2', bodyColor:'#8899bb',
          titleFont:{ family:'IBM Plex Mono', size:10 },
          bodyFont:{ family:'IBM Plex Mono', size:10 },
          callbacks:{
            title: items => new Date(items[0].parsed.x).toISOString().slice(0,10),
            label: item  => ` ${item.parsed.y?.toFixed(1)} m/yr`,
          }
        },
        annotation:{ annotations: buildAnnotations(lbl) },
      }
    }
  };
  const canvas = document.getElementById('chartCanvas');
  if (chart) chart.destroy();
  chart = new Chart(canvas, cfg);
  updateWindowPanel(lbl);
  updateROILabel(lbl);
  updateJSON();
}

function refreshAnnotations() {
  const lbl = roi_labels[currentIdx];
  chart.options.plugins.annotation.annotations = buildAnnotations(lbl);
  chart.update('none');
  updateWindowPanel(lbl);
  updateJSON();
}

function navigate(dir) {
  currentIdx = (currentIdx + dir + roi_labels.length) % roi_labels.length;
  dragState = null;
  renderChart(roi_labels[currentIdx]);
}

function updateROILabel(lbl) {
  const d = ALL_DATA[lbl];
  // Show typed label + shapefile ID to make any mismatch immediately visible
  document.getElementById('roi-label').textContent = d.type_label;
  const n = (windows[lbl] || []).length;
  document.getElementById('roi-info').textContent =
    `shapefile id ${d.orig_label} · ${n} window${n!==1?'s':''} selected`;
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') return;
  if (e.key==='ArrowRight'||e.key==='n') navigate(1);
  if (e.key==='ArrowLeft' ||e.key==='p') navigate(-1);
});

let dragState = null;
function getChartX(e) {
  const rect = chart.canvas.getBoundingClientRect();
  return chart.scales.x.getValueForPixel(e.clientX - rect.left);
}

document.getElementById('chartCanvas').addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  const t = getChartX(e);
  if (t === undefined) return;
  dragState = { start: t, current: t };
  setStatus('dragging...');
});

document.getElementById('chartCanvas').addEventListener('mousemove', e => {
  if (!dragState) return;
  dragState.current = getChartX(e);
  const lbl  = roi_labels[currentIdx];
  const t0   = Math.min(dragState.start, dragState.current);
  const t1   = Math.max(dragState.start, dragState.current);
  const anns = buildAnnotations(lbl);
  anns.drag  = { type:'box', xMin:t0, xMax:t1,
                 backgroundColor:'rgba(74,158,255,0.22)',
                 borderColor:'rgba(74,158,255,0.9)', borderWidth:1.5 };
  chart.options.plugins.annotation.annotations = anns;
  chart.update('none');
});

document.getElementById('chartCanvas').addEventListener('mouseup', e => {
  if (!dragState || e.button !== 0) return;
  const t1r = getChartX(e);
  const t0  = Math.min(dragState.start, t1r);
  const t1  = Math.max(dragState.start, t1r);
  dragState = null;
  if (t1 - t0 < 5*86400e3) { refreshAnnotations(); setStatus('ready'); return; }
  const lbl = roi_labels[currentIdx];
  windows[lbl].push([t0, t1]);
  save();
  setStatus(`added ${fmtDate(t0)} \u2192 ${fmtDate(t1)}`);
  refreshAnnotations();
  updateROILabel(lbl);
});

function updateWindowPanel(lbl) {
  const panel = document.getElementById('windows-panel');
  const wins  = windows[lbl] || [];
  if (!wins.length) {
    panel.innerHTML = '<span class="no-windows">No windows selected for this ROI</span>';
    return;
  }
  panel.innerHTML = wins.map((w, i) =>
    `<span class="win-tag" onclick="deleteWindow(${lbl},${i})" title="Click to delete">
      ${fmtDate(w[0])} \u2192 ${fmtDate(w[1])}<span class="del">✕</span>
    </span>`).join('');
}

function deleteWindow(lbl, idx) {
  windows[lbl].splice(idx, 1);
  save();
  refreshAnnotations();
  updateROILabel(roi_labels[currentIdx]);
  setStatus('deleted window');
}

function clearCurrentROI() {
  const lbl = roi_labels[currentIdx];
  windows[lbl] = [];
  save();
  refreshAnnotations();
  updateROILabel(lbl);
  setStatus('cleared');
}

// Flat list per ROI - plot_land-term_velocity_fig.py classifies by date automatically
function exportWindows() {
  const out = {};
  roi_labels.forEach(lbl => {
    if ((windows[lbl]||[]).length)
      out[lbl] = windows[lbl].map(w => [fmtDate(w[0]), fmtDate(w[1])]);
  });
  return out;
}

function updateJSON() {
  document.getElementById('json-out').value = JSON.stringify(exportWindows(), null, 2);
}

function save() {
  try { localStorage.setItem('vel_win_v3', JSON.stringify(exportWindows())); } catch(e) {}
  updateJSON();
}

function copyJSON() {
  const txt = document.getElementById('json-out').value;
  navigator.clipboard.writeText(txt).then(
    ()  => setStatus('\u2713 copied'),
    ()  => { document.getElementById('json-out').select(); document.execCommand('copy'); setStatus('\u2713 copied'); }
  );
}

function downloadJSON() {
  const blob = new Blob([document.getElementById('json-out').value], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'winter_windows.json';
  a.click();
}

const fmtDate  = ms  => new Date(ms).toISOString().slice(0,10);
const setStatus = msg => { document.getElementById('status').textContent = msg; };

renderChart(roi_labels[0]);
</script>
</body>
</html>
"""


# timeseries for each ROI, 2016-2025
print('Building data payload ...')
all_data = {}
roi_order = []
for lbl in config.ROI_LABELS:
    df = pd.read_csv(os.path.join(PREP_DIR, f'ROI_{lbl}_timeseries.csv'), parse_dates=['date'])
    df = df.sort_values('date').dropna(subset=['velocity', 'velocity_smooth'])
    df = df[(df['date'] >= '2016-01-01') & (df['date'] <= '2026-01-01')]
    all_data[str(lbl)] = {
        'orig_label': lbl,
        'type_label': f'L{lbl}',
        'times': [d.isoformat() for d in df['date']],
        'v_raw': [round(float(v), 4) for v in df['velocity']],
        'v_smooth': [round(float(v), 4) for v in df['velocity_smooth']],
        'v_err': [round(float(v), 4) for v in df['velocity_error']],
    }
    roi_order.append(lbl)
    print(f'  ROI {lbl} (L{lbl}): {len(df)} obs')

html = HTML.replace('__DATA_PLACEHOLDER__', json.dumps(all_data, separators=(',', ':')))
html = html.replace('__ORDER_PLACEHOLDER__', json.dumps(roi_order))
with open(OUTPUT_HTML, 'w') as f:
    f.write(html)
size_kb = os.path.getsize(OUTPUT_HTML) / 1024
print(f'\nWritten -> {OUTPUT_HTML}  ({size_kb:.0f} KB)')
print('Copy to your local PC and open in any browser.')
print(f'After selecting windows -> download winter_windows.json -> place in:\n  {PREP_DIR}/')
