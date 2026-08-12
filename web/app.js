/* Zone editor (PLAN.md section 6.6).
 *
 * Coordinates are normalised 0-1 everywhere in this file and only converted to canvas
 * pixels at draw time. That is what lets a zone survive a resolution change or a swap to
 * the camera's low-resolution substream.
 *
 * No framework: the whole editor is a canvas, a few forms, and fetch().
 */

const TYPES = {
  polygon:   { colour: '#50dc50', area: true,  label: 'Zone' },
  tripwire:  { colour: '#ffc800', area: false, label: 'Tripwire' },
  exclusion: { colour: '#ff5050', area: true,  label: 'Exclusion' },
  fire_roi:  { colour: '#ff8c00', area: true,  label: 'Fire ROI' },
};
const DAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'];
const HIT_RADIUS = 9;

const $ = (id) => document.getElementById(id);

const state = {
  mode: 'live',   // canvas engine vocabulary: live | edit | camera | events
  view: 'live',   // navigation vocabulary:    live | zones | camera | history
  zones: [],
  selected: -1,
  drawing: null,   // { type, points: [] }
  dragging: null,  // { zone, point }
  hover: null,
  dirty: false,
  days: new Set(DAYS),
};

const canvas = $('overlay');
const ctx = canvas.getContext('2d');
const backdrop = $('backdrop');

/* ------------------------------------------------------------------ utils */

function toast(message, kind = '') {
  const el = $('toast');
  el.textContent = message;
  el.className = `toast ${kind}`;
  el.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { el.hidden = true; }, 4200);
}

function pointerPos(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: clamp((event.clientX - rect.left) / rect.width),
    y: clamp((event.clientY - rect.top) / rect.height),
  };
}

const clamp = (v) => Math.min(1, Math.max(0, v));

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

/* ------------------------------------------------------------------- draw */

function draw() {
  const rect = canvas.getBoundingClientRect();
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);
  if (state.mode !== 'edit') return;

  state.zones.forEach((zone, index) => drawZone(zone, index === state.selected, W, H));

  if (state.drawing) {
    drawShape(state.drawing.points, TYPES[state.drawing.type], W, H, true, true);
  }
}

function drawZone(zone, selected, W, H) {
  const spec = TYPES[zone.type] || TYPES.polygon;
  drawShape(zone.points, spec, W, H, selected, false);

  if (zone.points.length) {
    const [px, py] = [zone.points[0][0] * W, zone.points[0][1] * H];
    ctx.fillStyle = spec.colour;
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.fillText(zone.name || zone.id, px + 6, py - 8);
  }
  if (!spec.area && zone.points.length >= 2) drawArrow(zone, spec, W, H);
}

function drawShape(points, spec, W, H, selected, active) {
  if (!points.length) return;
  const pts = points.map(([x, y]) => [x * W, y * H]);

  ctx.lineWidth = selected ? 3 : 2;
  ctx.strokeStyle = spec.colour;
  ctx.setLineDash(active ? [6, 4] : []);

  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));

  if (spec.area && pts.length >= 3) {
    ctx.closePath();
    ctx.fillStyle = spec.colour + '2e';
    ctx.fill();
  }
  ctx.stroke();
  ctx.setLineDash([]);

  if (selected || active) {
    pts.forEach(([x, y], i) => {
      ctx.beginPath();
      ctx.arc(x, y, i === state.hover?.point && selected ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = spec.colour;
      ctx.stroke();
    });
  }
}

function drawArrow(zone, spec, W, H) {
  /* Perpendicular arrow showing which crossing counts as ENTRY. */
  const [a, b] = [zone.points[0], zone.points[1]];
  const mid = [(a[0] + b[0]) / 2 * W, (a[1] + b[1]) / 2 * H];
  const dx = (b[0] - a[0]) * W, dy = (b[1] - a[1]) * H;
  const len = Math.hypot(dx, dy) || 1;

  const flip = zone.direction === 'b_to_a' ? -1 : 1;
  const nx = -dy / len * 26 * flip, ny = dx / len * 26 * flip;
  const tip = [mid[0] + nx, mid[1] + ny];

  ctx.strokeStyle = spec.colour;
  ctx.fillStyle = spec.colour;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(mid[0], mid[1]);
  ctx.lineTo(tip[0], tip[1]);
  ctx.stroke();

  const angle = Math.atan2(ny, nx);
  ctx.beginPath();
  ctx.moveTo(tip[0], tip[1]);
  ctx.lineTo(tip[0] - 9 * Math.cos(angle - 0.4), tip[1] - 9 * Math.sin(angle - 0.4));
  ctx.lineTo(tip[0] - 9 * Math.cos(angle + 0.4), tip[1] - 9 * Math.sin(angle + 0.4));
  ctx.closePath();
  ctx.fill();

  if (zone.direction === 'both') {
    ctx.beginPath();
    ctx.moveTo(mid[0], mid[1]);
    ctx.lineTo(mid[0] - nx, mid[1] - ny);
    ctx.stroke();
  }
}

/* -------------------------------------------------------------- hit tests */

function findVertex(pos) {
  const rect = canvas.getBoundingClientRect();
  for (let z = state.zones.length - 1; z >= 0; z--) {
    const points = state.zones[z].points;
    for (let p = 0; p < points.length; p++) {
      const dx = (points[p][0] - pos.x) * rect.width;
      const dy = (points[p][1] - pos.y) * rect.height;
      if (Math.hypot(dx, dy) <= HIT_RADIUS) return { zone: z, point: p };
    }
  }
  return null;
}

function findZone(pos) {
  for (let z = state.zones.length - 1; z >= 0; z--) {
    if (pointInPolygon(pos, state.zones[z].points)) return z;
  }
  return -1;
}

function pointInPolygon(pos, points) {
  if (points.length < 3) return false;
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i], [xj, yj] = points[j];
    if ((yi > pos.y) !== (yj > pos.y) &&
        pos.x < ((xj - xi) * (pos.y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/* ------------------------------------------------------------ interaction */

canvas.addEventListener('mousedown', (event) => {
  if (state.mode !== 'edit' || event.button !== 0) return;
  const pos = pointerPos(event);

  if (state.drawing) {
    state.drawing.points.push([pos.x, pos.y]);
    const spec = TYPES[state.drawing.type];
    // A two-point tripwire is complete as soon as the second point lands.
    if (!spec.area && state.drawing.points.length === 2) finishShape();
    updateHint();
    draw();
    return;
  }

  const vertex = findVertex(pos);
  if (vertex) {
    state.dragging = vertex;
    select(vertex.zone);
    return;
  }

  const zone = findZone(pos);
  if (zone >= 0) select(zone); else select(-1);
});

canvas.addEventListener('mousemove', (event) => {
  if (state.mode !== 'edit') return;
  const pos = pointerPos(event);

  if (state.dragging) {
    state.zones[state.dragging.zone].points[state.dragging.point] = [pos.x, pos.y];
    markDirty();
    draw();
    return;
  }
  const before = state.hover?.point;
  state.hover = findVertex(pos);
  canvas.style.cursor = state.hover ? 'grab' : (state.drawing ? 'crosshair' : 'default');
  if (before !== state.hover?.point) draw();
});

window.addEventListener('mouseup', () => { state.dragging = null; });

canvas.addEventListener('dblclick', (event) => {
  event.preventDefault();
  if (state.drawing) finishShape();
});

canvas.addEventListener('contextmenu', (event) => {
  if (state.mode !== 'edit') return;
  event.preventDefault();
  const vertex = findVertex(pointerPos(event));
  if (!vertex) return;

  const zone = state.zones[vertex.zone];
  const spec = TYPES[zone.type];
  const minimum = spec.area ? 3 : 2;
  if (zone.points.length <= minimum) {
    toast(`A ${spec.label.toLowerCase()} needs at least ${minimum} points.`, 'error');
    return;
  }
  zone.points.splice(vertex.point, 1);
  markDirty();
  draw();
});

window.addEventListener('keydown', (event) => {
  if (state.mode !== 'edit') return;
  if (event.key === 'Enter' && state.drawing) { event.preventDefault(); finishShape(); }
  if (event.key === 'Escape') { state.drawing = null; setTool(null); updateHint(); draw(); }
});

/* ---------------------------------------------------------------- drawing */

function setTool(type) {
  document.querySelectorAll('.tool').forEach((b) =>
    b.classList.toggle('active', b.dataset.type === type));
  $('btn-finish').disabled = !type;
  $('btn-cancel').disabled = !type;
  canvas.classList.toggle('drawing', Boolean(type));
}

document.querySelectorAll('.tool').forEach((button) => {
  button.addEventListener('click', () => {
    const type = button.dataset.type;
    state.drawing = { type, points: [] };
    select(-1);
    setTool(type);
    updateHint();
    draw();
  });
});

$('btn-finish').addEventListener('click', finishShape);
$('btn-cancel').addEventListener('click', () => {
  state.drawing = null; setTool(null); updateHint(); draw();
});

function finishShape() {
  if (!state.drawing) return;
  const { type, points } = state.drawing;
  const spec = TYPES[type];
  const minimum = spec.area ? 3 : 2;

  if (points.length < minimum) {
    toast(`A ${spec.label.toLowerCase()} needs at least ${minimum} points.`, 'error');
    return;
  }

  const zone = {
    id: uniqueId(type),
    type,
    name: '',
    points,
    detect: type === 'polygon' || type === 'tripwire' ? ['person'] : [],
    events: ['entry', 'exit'],
    severity: 'medium',
    direction: 'both',
  };
  if (!spec.area || type === 'exclusion' || type === 'fire_roi') {
    if (type === 'exclusion') zone.applies_to = ['fire', 'smoke'];
    if (type === 'fire_roi') { zone.applies_to = ['fire', 'smoke']; zone.conf_delta = -0.08; }
  }

  state.zones.push(zone);
  state.drawing = null;
  setTool(null);
  select(state.zones.length - 1);
  markDirty();
  renderList();
  updateHint();
  draw();
  validateSelected();
}

function uniqueId(type) {
  let n = 1;
  while (state.zones.some((z) => z.id === `${type}_${n}`)) n++;
  return `${type}_${n}`;
}

function updateHint() {
  const hint = $('hint');
  if (state.mode !== 'edit') { hint.textContent = ''; return; }
  if (state.drawing) {
    const spec = TYPES[state.drawing.type];
    hint.textContent = spec.area
      ? `Click to place corners · double-click or Enter to close · Esc to cancel (${state.drawing.points.length} placed)`
      : 'Click the two ends of the line · Esc to cancel';
  } else {
    hint.textContent = state.zones.length
      ? 'Click a boundary to select · drag a handle to move · right-click a handle to delete'
      : 'Choose a boundary type below to draw your first one';
  }
}

/* ------------------------------------------------------------------- list */

function renderList() {
  const list = $('zone-list');
  $('zone-count').textContent = state.zones.length;
  list.innerHTML = '';

  if (!state.zones.length) {
    list.innerHTML = '<li class="empty">No boundaries yet.</li>';
    return;
  }
  state.zones.forEach((zone, index) => {
    const li = document.createElement('li');
    li.className = index === state.selected ? 'selected' : '';
    li.innerHTML =
      `<i class="swatch ${zone.type}"></i>` +
      `<span class="zname">${escapeHtml(zone.name || zone.id)}</span>` +
      `<span class="ztype">${TYPES[zone.type]?.label ?? zone.type}</span>`;
    li.addEventListener('click', () => select(index));
    list.appendChild(li);
  });
}

const escapeHtml = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* ------------------------------------------------------------------ rules */

function select(index) {
  state.selected = index;
  renderList();
  const panel = $('rules-panel');
  panel.hidden = index < 0 || state.view !== 'zones';
  if (index >= 0) loadRules(state.zones[index]);
  draw();
}

function loadRules(zone) {
  const spec = TYPES[zone.type];
  $('f-name').value = zone.name || '';
  $('f-severity').value = zone.severity || 'medium';
  $('f-minframes').value = zone.min_frames ?? '';
  $('f-direction').value = zone.direction || 'both';
  $('f-confdelta').value = zone.conf_delta ?? -0.08;

  const emitting = zone.type === 'polygon' || zone.type === 'tripwire';
  $('row-events').hidden = !emitting;
  $('row-severity').hidden = !emitting;
  $('row-minframes').hidden = !emitting;
  $('row-direction').hidden = zone.type !== 'tripwire';
  $('row-confdelta').hidden = zone.type !== 'fire_roi';

  const classes = emitting ? (zone.detect || []) : (zone.applies_to || []);
  $('f-detect').querySelectorAll('input').forEach((cb) => {
    cb.checked = classes.includes(cb.value);
  });
  $('f-events').querySelectorAll('input').forEach((cb) => {
    cb.checked = (zone.events || []).includes(cb.value);
  });

  const schedule = zone.schedule;
  const restricted = Boolean(schedule && schedule.from);
  $('f-sched-on').checked = restricted;
  $('sched-fields').hidden = !restricted;
  if (schedule) {
    if (schedule.from) $('f-sched-from').value = schedule.from;
    if (schedule.to) $('f-sched-to').value = schedule.to;
    state.days = new Set(schedule.days || DAYS);
  } else {
    state.days = new Set(DAYS);
  }
  renderDays();
  updateScheduleNote();
  void spec;
}

function collectRules() {
  const index = state.selected;
  if (index < 0) return;
  const zone = state.zones[index];
  const emitting = zone.type === 'polygon' || zone.type === 'tripwire';

  zone.name = $('f-name').value.trim();
  const checked = [...$('f-detect').querySelectorAll('input:checked')].map((c) => c.value);
  if (emitting) zone.detect = checked; else zone.applies_to = checked;

  if (emitting) {
    zone.events = [...$('f-events').querySelectorAll('input:checked')].map((c) => c.value);
    zone.severity = $('f-severity').value;
    const minFrames = $('f-minframes').value;
    if (minFrames) zone.min_frames = Number(minFrames); else delete zone.min_frames;
  }
  if (zone.type === 'tripwire') zone.direction = $('f-direction').value;
  if (zone.type === 'fire_roi') zone.conf_delta = Number($('f-confdelta').value);

  if ($('f-sched-on').checked) {
    zone.schedule = {
      days: DAYS.filter((d) => state.days.has(d)),
      from: $('f-sched-from').value,
      to: $('f-sched-to').value,
    };
  } else {
    delete zone.schedule;
  }

  markDirty();
  renderList();
  draw();
}

['f-name', 'f-severity', 'f-minframes', 'f-direction', 'f-confdelta',
 'f-sched-from', 'f-sched-to'].forEach((id) => {
  $(id).addEventListener('input', collectRules);
  $(id).addEventListener('change', collectRules);
});
['f-detect', 'f-events'].forEach((id) =>
  $(id).addEventListener('change', collectRules));

$('f-sched-on').addEventListener('change', (event) => {
  $('sched-fields').hidden = !event.target.checked;
  collectRules();
  updateScheduleNote();
});

function renderDays() {
  const wrap = $('f-days');
  wrap.innerHTML = '';
  DAYS.forEach((day) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = day;
    button.className = state.days.has(day) ? 'on' : '';
    button.addEventListener('click', () => {
      if (state.days.has(day)) state.days.delete(day); else state.days.add(day);
      renderDays();
      collectRules();
    });
    wrap.appendChild(button);
  });
}

function updateScheduleNote() {
  const from = $('f-sched-from').value, to = $('f-sched-to').value;
  $('sched-note').textContent = from && to && from > to
    ? 'Overnight window — the morning half belongs to the previous day.'
    : '';
}

$('btn-delete').addEventListener('click', () => {
  if (state.selected < 0) return;
  state.zones.splice(state.selected, 1);
  select(-1);
  markDirty();
  renderList();
  draw();
});

/* --------------------------------------------------------------- validate */

async function validateSelected() {
  if (state.selected < 0) return;
  const zone = state.zones[state.selected];
  try {
    const response = await fetch('/api/zones/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: zone.type, points: zone.points }),
    });
    const result = await response.json();
    if (!result.valid) toast(result.errors.map((e) => e.message).join('; '), 'error');
    else if (result.warnings?.length) toast(result.warnings.join('; '));
  } catch { /* validation is advisory; save is authoritative */ }
}

/* ------------------------------------------------------------------- save */

function markDirty() {
  state.dirty = true;
  const save = $('btn-save');
  if (save) save.disabled = false;
}

async function saveZones() {
  collectRules();
  const save = $('btn-save');
  if (save) save.disabled = true;
  try {
    const response = await fetch('/api/zones', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zones: state.zones }),
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail || `save failed (${response.status})`);
    }
    const result = await response.json();
    state.dirty = false;
    const noun = result.saved === 1 ? 'boundary' : 'boundaries';
    toast(`Saved ${result.saved} ${noun}. Applied without a restart.`, 'ok');
  } catch (error) {
    if (save) save.disabled = false;
    toast(error.message, 'error');
  }
}

window.addEventListener('beforeunload', (event) => {
  if (state.dirty) { event.preventDefault(); event.returnValue = ''; }
});

/* ----------------------------------------------------------------- camera */

let cameraLoaded = false;

function setCctv(useCctv) {
  document.querySelectorAll('#f-use-cctv button').forEach((b) =>
    b.classList.toggle('on', (b.dataset.cctv === 'true') === useCctv));
  $('row-webcam').hidden = useCctv;
  $('row-rtsp').hidden = !useCctv;
}

document.querySelectorAll('#f-use-cctv button').forEach((button) => {
  button.addEventListener('click', () => setCctv(button.dataset.cctv === 'true'));
});

function updateCadenceNote() {
  const decode = Number($('f-decode').value) || 0;
  // person_every_n is server-side tuning; 2 is the shipped default and what the note
  // is meant to convey - the boundary hysteresis is counted in THESE frames, not
  // camera frames.
  const detection = decode / 2;
  $('cadence-note').textContent = decode
    ? `Detection runs at ~${detection.toFixed(1)} Hz. `
      + 'Boundary hysteresis is counted in these frames, not camera frames.'
    : '';
}
$('f-decode').addEventListener('input', updateCadenceNote);

async function loadCamera() {
  try {
    const cam = await (await fetch('/api/camera')).json();
    setCctv(cam.use_cctv);
    $('f-source').value = cam.source ?? '0';
    $('f-camid').value = cam.id ?? 'cam_01';
    $('f-width').value = cam.width;
    $('f-height').value = cam.height;
    $('f-fps').value = cam.fps;
    $('f-decode').value = cam.decode_fps;
    $('f-autostart').checked = Boolean(cam.autostart);

    // The URL is never sent to the browser. Say whether one is stored, and leave the
    // field blank so a save without typing keeps what is already there.
    $('f-rtsp').value = '';
    $('rtsp-note').textContent = cam.rtsp_url_set
      ? `Stored: ${cam.rtsp_url_redacted}. Leave blank to keep it, or type a new URL to replace it.`
      : 'No URL stored yet.';

    if (cam.actual_size && cam.actual_size[0]) {
      const [w, h] = cam.actual_size;
      if (w !== cam.width || h !== cam.height) {
        $('rtsp-note').textContent +=
          ` Camera is actually streaming ${w}x${h}.`;
      }
    }
    updateCadenceNote();
    cameraLoaded = true;
  } catch { toast('Could not load camera settings.', 'error'); }
}

function cameraPayload() {
  const useCctv = document.querySelector('#f-use-cctv button.on')?.dataset.cctv === 'true';
  const payload = {
    use_cctv: useCctv,
    source: $('f-source').value.trim() || '0',
    id: $('f-camid').value.trim() || 'cam_01',
    width: $('f-width').value,
    height: $('f-height').value,
    fps: $('f-fps').value,
    decode_fps: $('f-decode').value,
    autostart: $('f-autostart').checked,
  };
  const rtsp = $('f-rtsp').value.trim();
  if (rtsp) payload.rtsp_url = rtsp;   // omitted = keep the stored one
  return payload;
}

function showTestResult(result) {
  const box = $('test-result');
  box.hidden = false;
  if (!result.ok) {
    box.className = 'test-result error';
    box.textContent = result.error || 'Connection failed.';
    return;
  }
  box.className = 'test-result ok';
  const fps = result.reported_fps ? ` · reports ${result.reported_fps} fps` : '';
  box.innerHTML =
    `Connected in ${result.elapsed_ms} ms — <code>${escapeHtml(result.source)}</code><br>` +
    `Actual frame: <code>${result.actual_width}x${result.actual_height}</code>${fps}` +
    (result.warnings || []).map((w) => `<span class="warn">⚠ ${escapeHtml(w)}</span>`).join('');
}

$('btn-test').addEventListener('click', async () => {
  const button = $('btn-test');
  button.disabled = true;
  button.textContent = 'Testing…';
  try {
    const response = await fetch('/api/camera/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cameraPayload()),
    });
    showTestResult(await response.json());
  } catch (error) {
    showTestResult({ ok: false, error: error.message });
  } finally {
    button.disabled = false;
    button.textContent = 'Test connection';
  }
});

$('btn-camera-save').addEventListener('click', async () => {
  const button = $('btn-camera-save');
  button.disabled = true;
  try {
    const response = await fetch('/api/camera', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cameraPayload()),
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail || `save failed (${response.status})`);
    }
    const result = await response.json();
    toast(`Camera saved — reconnecting to ${result.resolved}`, 'ok');
    $('f-rtsp').value = '';
    await loadCamera();
    if (state.mode === 'live') backdrop.src = `/api/stream?t=${Date.now()}`;
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    button.disabled = false;
  }
});

/* ---------------------------------------------------------------- history */

const KIND_LABEL = { boundary: 'Boundary', fire: 'Fire', smoke: 'Smoke', health: 'Camera' };

function formatWhen(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  // Local time with seconds: an alert log is read against wall-clock memory
  // ("what happened just before 2am"), so the viewer's own zone is the useful one.
  return d.toLocaleString(undefined, {
    day: '2-digit', month: 'short', hour: '2-digit',
    minute: '2-digit', second: '2-digit', hour12: false,
  });
}

function pillClass(event) {
  if (event.kind === 'boundary') return event.subtype === 'entry' ? 'entry' : 'exit';
  return event.kind || 'health';
}

function renderStats(summary, events) {
  const sum = (o) => Object.values(o).reduce((a, b) => a + b, 0);
  let today;
  let total;
  let suffix = '';

  if (summary.available) {
    today = summary.today || {};
    total = summary.counts || {};
  } else {
    // Without PostgreSQL there is nothing to aggregate, but showing "0 today" above a
    // table listing today's alerts is a flat contradiction an operator would rightly
    // distrust. Count what is actually on screen and label it as partial.
    const midnight = new Date();
    midnight.setHours(0, 0, 0, 0);
    today = {};
    for (const e of events) {
      if (new Date(e.ts) >= midnight) today[e.kind] = (today[e.kind] || 0) + 1;
    }
    total = today;
    suffix = ' (in memory)';
  }

  const cells = [
    ['Today' + suffix, sum(today)],
    ['Boundary today', today.boundary || 0],
    ['Fire / smoke today', (today.fire || 0) + (today.smoke || 0)],
    [summary.available ? 'All time' : 'Held in memory', sum(total)],
  ];
  $('h-stats').innerHTML = cells
    .map(([k, n]) => `<div class="stat"><div class="n">${n}</div><div class="k">${k}</div></div>`)
    .join('');
}

async function loadHistory() {
  const kind = $('h-kind').value;
  const zone = $('h-zone').value;
  const limit = $('h-limit').value;

  const params = new URLSearchParams({ limit });
  if (kind) params.set('kind', kind);
  if (zone) params.set('zone_id', zone);

  try {
    const [data, summary] = await Promise.all([
      (await fetch(`/api/events?${params}`)).json(),
      (await fetch('/api/events/summary')).json(),
    ]);

    renderStats(summary, data.events);
    const rows = $('h-rows');

    if (!data.events.length) {
      rows.innerHTML = '<tr><td colspan="6" class="empty">No alerts recorded yet.</td></tr>';
    } else {
      rows.innerHTML = data.events.map((e) => `
        <tr>
          <td class="when">${escapeHtml(formatWhen(e.ts))}</td>
          <td class="msg">${escapeHtml(e.message || '')}</td>
          <td>${escapeHtml(e.zone_name || e.zone_id || '—')}</td>
          <td><span class="pill ${pillClass(e)}">${escapeHtml(
            e.subtype || KIND_LABEL[e.kind] || e.kind)}</span></td>
          <td>${e.track_id != null ? '#' + e.track_id : '—'}</td>
          <td class="sev-${escapeHtml(e.severity || 'medium')}">${escapeHtml(e.severity || '—')}</td>
        </tr>`).join('');
    }

    $('h-source').textContent = data.source === 'postgres'
      ? 'Stored in PostgreSQL.'
      : 'PostgreSQL unavailable — showing recent alerts held in memory only. '
        + 'These are NOT being recorded.';
  } catch {
    toast('Could not load alert history.', 'error');
  }
}

function refreshZoneFilter() {
  const select = $('h-zone');
  const current = select.value;
  select.innerHTML = '<option value="">All boundaries</option>' +
    state.zones.filter((z) => z.type === 'polygon' || z.type === 'tripwire')
      .map((z) => `<option value="${escapeHtml(z.id)}">${escapeHtml(z.name || z.id)}</option>`)
      .join('');
  select.value = current;
}

['h-kind', 'h-zone', 'h-limit'].forEach((id) => $(id).addEventListener('change', loadHistory));
$('h-refresh').addEventListener('click', loadHistory);

/* ------------------------------------------------------------------ views */

const VIEWS = {
  live:    { title: 'Live view',     sub: 'Detections and boundaries as they happen.' },
  zones:   { title: 'Boundaries',    sub: 'Draw the areas and lines that raise alerts.' },
  camera:  { title: 'Camera',        sub: 'Source, resolution and capture rate.' },
  history: { title: 'Alert history', sub: 'Every alert this system has recorded.' },
};

function renderTopbarActions(view) {
  const host = $('topbar-actions');
  host.innerHTML = '';
  if (view !== 'zones') return;

  const save = document.createElement('button');
  save.id = 'btn-save';
  save.className = 'primary';
  save.textContent = 'Save boundaries';
  save.disabled = !state.dirty;
  save.addEventListener('click', saveZones);
  host.appendChild(save);
}

function setView(view) {
  // `state.mode` keeps the drawing engine's vocabulary ('edit'), while the nav uses the
  // operator's ('zones'). Renaming inside the canvas code would touch far more than it
  // is worth.
  state.mode = view === 'zones' ? 'edit' : view === 'history' ? 'events' : view;
  state.view = view;

  document.querySelectorAll('.nav-item').forEach((b) =>
    b.classList.toggle('active', b.dataset.view === view));

  $('view-title').textContent = VIEWS[view].title;
  $('view-sub').textContent = VIEWS[view].sub;
  renderTopbarActions(view);

  $('view-video').hidden = view !== 'live' && view !== 'zones';
  $('view-camera').hidden = view !== 'camera';
  $('view-history').hidden = view !== 'history';

  $('toolbar').hidden = view !== 'zones';
  $('zones-panel').hidden = view !== 'zones';
  $('recent-panel').hidden = view !== 'live';
  if (view !== 'zones') $('rules-panel').hidden = true;

  if (view === 'history') {
    refreshZoneFilter();
    loadHistory();
    return;
  }
  if (view === 'camera') {
    if (!cameraLoaded) loadCamera();
    return;
  }

  if (view === 'zones') {
    // Freeze a still as the drawing backdrop: a moving image makes precise clicking
    // miserable, and the boundary is static anyway.
    backdrop.src = `/api/snapshot?t=${Date.now()}`;
    select(state.selected);
  } else {
    state.drawing = null;
    setTool(null);
    backdrop.src = `/api/stream?t=${Date.now()}`;
  }
  updateHint();
  draw();
}

document.querySelectorAll('.nav-item').forEach((button) => {
  button.addEventListener('click', () => setView(button.dataset.view));
});

/* ----------------------------------------------------------------- polling */

async function loadZones() {
  try {
    const data = await (await fetch('/api/zones')).json();
    if (state.dirty) return;
    state.zones = data.zones || [];
    if (data.error) toast(`Boundary file rejected — previous config kept: ${data.error}`, 'error');
    renderList();
    updateHint();
    draw();
  } catch { toast('Cannot reach the API.', 'error'); }
}

async function pollHealth() {
  try {
    const health = await (await fetch('/api/health')).json();
    const live = health.feed_state === 'live';
    $('feed-dot').className = `dot ${live ? 'live' : health.feed_state === 'lost' ? 'lost' : ''}`;
    $('feed-text').textContent = health.feed_state || 'unknown';

    // Storage reported separately from the camera. "Alerts firing but not recorded" is
    // a different problem from "site unwatched", and needs a different response.
    const db = health.database || {};
    $('db-dot').className = `dot ${db.connected ? 'live' : db.configured ? 'lost' : ''}`;
    $('db-text').textContent = db.connected
      ? 'recording'
      : db.configured ? 'db down' : 'not recording';

    $('sys-zones').textContent = health.zones ?? 0;
    $('sys-perf').textContent = health.inference_ms != null
      ? `${health.render_fps ?? 0} fps · ${health.inference_ms} ms · `
        + `${health.people_tracked ?? 0} tracked · ${health.events_fired ?? 0} alerts`
      : 'pipeline not running';
  } catch {
    $('feed-dot').className = 'dot lost';
    $('feed-text').textContent = 'unreachable';
    $('sys-perf').textContent = 'API unreachable';
  }
}

async function pollEvents() {
  try {
    const data = await (await fetch('/api/events?limit=20')).json();
    const list = $('events');
    list.innerHTML = data.events.length
      ? ''
      : '<li class="empty">Nothing yet.</li>';
    data.events.forEach((event) => {
      const li = document.createElement('li');
      const when = new Date(event.ts).toLocaleTimeString(undefined, { hour12: false });
      li.innerHTML = `<span class="when">${escapeHtml(when)}</span>` +
        `<span class="${escapeHtml(event.subtype || event.kind)}">` +
        `${escapeHtml(event.message || '')}</span>`;
      list.appendChild(li);
    });
  } catch { /* transient */ }
}

/* ------------------------------------------------------------------- init */

backdrop.addEventListener('load', resizeCanvas);
window.addEventListener('resize', resizeCanvas);

renderDays();
loadZones();
setView('live');
pollHealth();
pollEvents();
setInterval(pollHealth, 2000);
setInterval(pollEvents, 3000);
setInterval(() => { if (!state.dirty) loadZones(); }, 10000);
