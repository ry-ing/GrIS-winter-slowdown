# HTML tool for picking the winter windows from timeseries 

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
<style>
body{font-family:sans-serif;font-size:13px;margin:10px}
.chart-wrap{height:500px}
#chartCanvas{cursor:crosshair}
.win-tag{cursor:pointer;border:1px solid #888;padding:2px 6px;margin-right:6px}
#json-out{width:100%;height:80px}
</style>
</head>
<body>
<p>Click-drag to mark a winter window. Click a window below the plot to delete it.
Arrow keys (or p/n) change ROI. Baseline / W1-W4 classification is automatic in the plot script.</p>
<div>
  <button onclick="navigate(-1)">&lt;</button>
  <b id="roi-label"></b>
  <button onclick="navigate(1)">&gt;</button>
  <span id="roi-info"></span>
  <button onclick="clearCurrentROI()">Clear ROI</button>
  <span id="status">ready</span>
</div>
<p>Blue line: smoothed, grey dots: raw, dashed line: Aug 2021 event, shaded: selected windows</p>
<div class="chart-wrap"><canvas id="chartCanvas"></canvas></div>
<div id="windows-panel">
  No windows selected for this ROI
</div>
<div>
  <button onclick="copyJSON()">Copy JSON</button>
  <button onclick="downloadJSON()">Download JSON</button>
  <textarea id="json-out" readonly placeholder="JSON will appear here..."></textarea>
</div>

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
    borderColor: 'black', borderWidth: 1.5, borderDash: [5, 4],
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
      { label:'Smoothed', data:smooth, borderColor:'dodgerblue', borderWidth:2,
        pointRadius:0, tension:0.25, order:1 },
      { label:'Raw', data:raw, borderColor:'transparent', backgroundColor:'grey',
        pointRadius:2.2, pointStyle:'circle', showLine:false, order:2 },
    ]},
    options: {
      responsive:true, maintainAspectRatio:false,
      animation:{ duration:100 }, parsing:false,
      scales: {
        x: {
          type:'time', min:X_MIN, max:X_MAX,
          time:{ unit:'year', displayFormats:{ year:'yyyy' } },
          ticks:{ maxTicksLimit:11 },
        },
        y: {
          title:{ display:true, text:'Ice speed (m yr\u207b\u00b9)' },
        },
      },
      plugins: {
        legend:{ display:false },
        tooltip:{
          mode:'nearest', intersect:false,
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
    panel.innerHTML = 'No windows selected for this ROI';
    return;
  }
  panel.innerHTML = wins.map((w, i) =>
    `<span class="win-tag" onclick="deleteWindow(${lbl},${i})" title="Click to delete">
      ${fmtDate(w[0])} \u2192 ${fmtDate(w[1])} x
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
    ()  => setStatus('copied'),
    ()  => { document.getElementById('json-out').select(); document.execCommand('copy'); setStatus('copied'); }
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
