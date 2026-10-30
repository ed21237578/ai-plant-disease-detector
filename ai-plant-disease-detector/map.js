/* PDD field map (Person 3).
 * Draws every saved scan as a coloured pin, splits the field into numbered zones and says which zones to treat.
 *
 * Needs: Leaflet (vendor/leaflet/leaflet.js + leaflet.css), storage.js (getScans), an element <div id="map">
 *        and optionally <div id="map-summary"> for the treatment plan.
 * Listens: 'pdd:scans-changed', 'pdd:scan-saved' (redraw), 'pdd:show-map' (map became visible).
 * Colours: window.scanStatus(scan) from scan.js if present, else the same rule here:
 *          confidence < 80% -> 'check' (amber), healthy label -> 'healthy' (green), otherwise 'treat' (red).
 */
(function () {
  'use strict';

  // ---------- settings ----------
  const CONF_T = 0.80;
  const CELL_M = 20;            // smallest zone size in metres
  const MAX_CELLS = 10;         // at most 10 x 10 zones; zones grow for bigger fields
  const PAD_M = 6;              // margin around the outermost scans
  // Demo field centre. Change this to a real field (right-click it in Google Maps to copy its coordinates).
  const DEMO_CENTER = [52.8357, -0.0662];
  const COLORS = { healthy: '#3b8a48', check: '#e0a100', treat: '#c0392b', none: 'transparent' };
  const RANK = { none: 0, healthy: 1, check: 2, treat: 3 };

  const mapEl = document.getElementById('map');
  const sumEl = document.getElementById('map-summary');
  if (!mapEl) return;

  // ---------- styles for the map + summary (kept here so no other file needs editing) ----------
  const css = `
  #map { min-height: 320px; }
  .pdd-zone-num { color: #fff; font: 700 11px/1 system-ui, sans-serif; text-shadow: 0 0 3px #000, 0 0 2px #000; text-align: left; pointer-events: none; }
  .pdd-pop { display: grid; gap: 6px; min-width: 170px; font: 14px/1.35 system-ui, sans-serif; }
  .pdd-pop img { width: 72px; height: 72px; object-fit: cover; border-radius: 8px; }
  .pdd-pop b { font-size: 15px; }
  .pdd-pop small { color: #555; }
  .pdd-pop button { font: inherit; font-size: 13px; padding: 5px 9px; border-radius: 8px; border: 1px solid #ccc; background: #fff; cursor: pointer; justify-self: start; }
  .pdd-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: 0; }
  #map-summary { display: grid; gap: 12px; margin-top: 4px; }
  .pdd-sum-card { background: var(--surface, #fff); border: 1px solid var(--line, #d5dfd2); border-radius: 16px; padding: 16px; display: grid; gap: 12px; color: var(--ink, #17241a); }
  .pdd-headline { font-size: 20px; font-weight: 800; line-height: 1.25; margin: 0; }
  .pdd-headline.treat { color: var(--treat, #c0392b); }
  .pdd-headline.ok { color: var(--ok, #3b8a48); }
  .pdd-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
  .pdd-stat { background: var(--surface-2, #f5f8f3); border-radius: 12px; padding: 10px; }
  .pdd-stat b { display: block; font-size: 24px; line-height: 1.1; font-variant-numeric: tabular-nums; }
  .pdd-stat span { font-size: 12px; font-weight: 600; color: var(--muted, #56675a); }
  .pdd-zones { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
  .pdd-zones li { display: grid; grid-template-columns: 52px minmax(0, 1fr); gap: 10px; align-items: start; background: var(--surface-2, #f5f8f3); border-radius: 10px; padding: 10px; font-size: 14px; }
  .pdd-zones .z { font-weight: 800; text-align: center; border-radius: 7px; padding: 3px 0; color: #fff; font-variant-numeric: tabular-nums; }
  .pdd-zones small { display: block; color: var(--muted, #56675a); }
  .pdd-legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 13px; color: var(--muted, #56675a); }
  .pdd-note { font-size: 13px; color: var(--muted, #56675a); margin: 0; }
  .pdd-btns { display: flex; flex-wrap: wrap; gap: 8px; }
  .pdd-btns button { font: inherit; font-weight: 700; font-size: 14px; padding: 10px 14px; min-height: 44px; border-radius: 12px; border: 1px solid var(--line, #d5dfd2); background: var(--surface, #fff); color: var(--ink, #17241a); cursor: pointer; }
  .pdd-btns button.primary { background: var(--leaf, #2d6a39); color: var(--leaf-ink, #fff); border-color: var(--leaf, #2d6a39); }
  `;
  const styleTag = document.createElement('style');
  styleTag.textContent = css;
  document.head.appendChild(styleTag);

  // ---------- helpers ----------
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (p) => Math.round(p * 100) + '%';
  const hasLoc = (s) => typeof s.lat === 'number' && typeof s.lng === 'number' && isFinite(s.lat) && isFinite(s.lng);
  const isHealthy = (label) => /___healthy$/i.test(label || '');
  function statusOf(s) {
    if (typeof window.scanStatus === 'function') return window.scanStatus(s);
    if (typeof s.confidence !== 'number' || s.confidence < CONF_T) return 'check';
    return isHealthy(s.label) ? 'healthy' : 'treat';
  }
  function joinList(nums) {
    if (nums.length <= 1) return nums.join('');
    return nums.slice(0, -1).join(', ') + ' and ' + nums[nums.length - 1];
  }
  let treatments = {};
  if (window.PDD_TREATMENTS_DATA) treatments = window.PDD_TREATMENTS_DATA;
  else fetch('treatments.json').then((r) => (r.ok ? r.json() : {})).then((t) => { treatments = t || {}; render(); }).catch(() => {});
  function niceName(label) {
    const t = treatments[label];
    if (t && t.name) return t.name;
    const [crop, cond] = String(label || '').split('___');
    const c = (crop || '').replace(/,_bell/i, ' (bell)').replace(/_/g, ' ');
    const d = (cond || '').replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
    return d ? `${c} · ${d.charAt(0).toUpperCase()}${d.slice(1)}` : c;
  }

  // ---------- map ----------
  if (typeof window.L === 'undefined') {
    mapEl.innerHTML = '<p style="padding:16px">The map library did not load. Check that vendor/leaflet/leaflet.js is in the project.</p>';
    return;
  }
  if (!mapEl.clientHeight) mapEl.style.height = '60vh';

  const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxNativeZoom: 19, maxZoom: 21, attribution: 'Imagery © Esri, Maxar',
  });
  const street = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxNativeZoom: 19, maxZoom: 21, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  });
  const map = L.map(mapEl, { layers: [satellite], zoomControl: true, zoomSnap: 0.25 }).setView(DEMO_CENTER, 17);
  L.control.layers({ Satellite: satellite, Street: street }, null, { position: 'topright' }).addTo(map);
  L.control.scale({ imperial: false }).addTo(map);
  const gridLayer = L.layerGroup().addTo(map);
  const pinLayer = L.layerGroup().addTo(map);
  let lastFieldKey = '';

  // ---------- grid ----------
  function buildGrid(scans) {
    const lats = scans.map((s) => s.lat), lngs = scans.map((s) => s.lng);
    const minLat = Math.min(...lats), maxLat = Math.max(...lats), minLng = Math.min(...lngs), maxLng = Math.max(...lngs);
    const cLat = (minLat + maxLat) / 2, cLng = (minLng + maxLng) / 2;
    const mLat = 111320, mLng = 111320 * Math.cos((cLat * Math.PI) / 180);
    const wM = (maxLng - minLng) * mLng + 2 * PAD_M;
    const hM = (maxLat - minLat) * mLat + 2 * PAD_M;
    const cell = Math.ceil(Math.max(CELL_M, wM / MAX_CELLS, hM / MAX_CELLS) / 5) * 5;
    const cols = Math.max(1, Math.ceil(wM / cell)), rows = Math.max(1, Math.ceil(hM / cell));
    const west = cLng - (cols * cell) / 2 / mLng, north = cLat + (rows * cell) / 2 / mLat;
    const cells = [];
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      cells.push({
        n: r * cols + c + 1, r, c, scans: [], status: 'none',
        bounds: [[north - ((r + 1) * cell) / mLat, west + (c * cell) / mLng], [north - (r * cell) / mLat, west + ((c + 1) * cell) / mLng]],
      });
    }
    for (const s of scans) {
      const c = Math.min(cols - 1, Math.max(0, Math.floor(((s.lng - west) * mLng) / cell)));
      const r = Math.min(rows - 1, Math.max(0, Math.floor(((north - s.lat) * mLat) / cell)));
      const cellObj = cells[r * cols + c];
      cellObj.scans.push(s);
      const st = statusOf(s);
      if (RANK[st] > RANK[cellObj.status]) cellObj.status = st;
    }
    return { cells, rows, cols, cell, bounds: [[north - (rows * cell) / mLat, west], [north, west + (cols * cell) / mLng]] };
  }

  function popupHtml(s) {
    const st = statusOf(s);
    const label = st === 'check' ? `Not sure (best guess: ${esc(niceName(s.label))})` : esc(niceName(s.label));
    const when = s.time ? new Date(s.time).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '';
    return `<div class="pdd-pop">
      ${s.thumb ? `<img src="${esc(s.thumb)}" alt="">` : ''}
      <b><span class="pdd-dot" style="background:${COLORS[st]}"></span>${label}</b>
      <span>${typeof s.confidence === 'number' ? pct(s.confidence) + ' confident' : ''}</span>
      <small>${esc(when)}${s.accuracy ? ` · GPS ±${esc(s.accuracy)} m` : ''}${s.demo ? ' · demo scan' : ''}</small>
      <button type="button" data-del="${esc(s.id)}">Delete scan</button>
    </div>`;
  }

  // ---------- render ----------
  let busy = false, again = false;
  async function render() {
    if (busy) { again = true; return; }
    busy = true;
    try { do { again = false; await renderOnce(); } while (again); } finally { busy = false; }
  }
  async function renderOnce() {
    {
      const all = typeof window.getScans === 'function' ? await window.getScans() : [];
      const located = all.filter(hasLoc);
      gridLayer.clearLayers();
      pinLayer.clearLayers();
      let grid = null;
      if (located.length) {
        grid = buildGrid(located);
        for (const cell of grid.cells) {
          L.rectangle(cell.bounds, {
            color: '#ffffff', weight: 1, opacity: 0.8,
            fillColor: COLORS[cell.status], fillOpacity: cell.status === 'none' ? 0 : cell.status === 'healthy' ? 0.28 : 0.45,
            interactive: false,
          }).addTo(gridLayer);
          L.marker([cell.bounds[1][0], cell.bounds[0][1]], { // north-west corner of the zone
            interactive: false, keyboard: false,
            icon: L.divIcon({ className: 'pdd-zone-num', html: String(cell.n), iconSize: [24, 14], iconAnchor: [-3, -3] }),
          }).addTo(gridLayer);
        }
        for (const s of located) {
          L.circleMarker([s.lat, s.lng], { radius: 8, color: '#ffffff', weight: 2, fillColor: COLORS[statusOf(s)], fillOpacity: 1 })
            .bindPopup(popupHtml(s)).addTo(pinLayer);
        }
        const key = grid.bounds.flat().map((v) => v.toFixed(6)).join(',');
        if (key !== lastFieldKey) { lastFieldKey = key; fitField(grid); }
      }
      renderSummary(all, located, grid);
    }
  }

  let lastGrid = null;
  function fitField(grid) {
    lastGrid = grid;
    if (mapEl.clientWidth) map.fitBounds(grid.bounds, { padding: [20, 20], maxZoom: 19 });
  }

  function renderSummary(all, located, grid) {
    if (!sumEl) return;
    const demoCount = all.filter((s) => s.demo).length;
    const noLoc = all.length - located.length;
    let html = '<div class="pdd-sum-card">';
    if (!grid) {
      html += `<p class="pdd-headline">No scans on the map yet</p>
        <p class="pdd-note">Scan a leaf and press “Save to field map”. Each scan is pinned where you are standing.${noLoc ? ` ${noLoc} saved scan${noLoc > 1 ? 's have' : ' has'} no GPS location yet.` : ''}</p>`;
    } else {
      const total = grid.cells.length;
      const treat = grid.cells.filter((c) => c.status === 'treat');
      const check = grid.cells.filter((c) => c.status === 'check');
      const scanned = grid.cells.filter((c) => c.status !== 'none');
      const notSprayed = Math.round(((total - treat.length) / total) * 100);
      html += treat.length
        ? `<p class="pdd-headline treat">Treat zone${treat.length > 1 ? 's' : ''} ${joinList(treat.map((c) => c.n))} only</p>`
        : `<p class="pdd-headline ok">Nothing to treat${check.length ? ' yet' : ''}</p>`;
      html += `<div class="pdd-stats">
          <div class="pdd-stat"><b>${notSprayed}%</b><span>of field not sprayed</span></div>
          <div class="pdd-stat"><b>${treat.length}/${total}</b><span>zones to treat</span></div>
          <div class="pdd-stat"><b>${located.length}</b><span>scans mapped</span></div>
        </div>`;
      const items = [...treat, ...check];
      if (items.length) {
        html += '<ul class="pdd-zones">';
        for (const c of items) {
          const counts = {};
          c.scans.filter((s) => statusOf(s) === c.status).forEach((s) => { counts[s.label] = (counts[s.label] || 0) + 1; });
          const what = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([l, n]) => `${esc(niceName(l))}${n > 1 ? ` ×${n}` : ''}`).join(', ');
          const t = treatments[Object.keys(counts)[0]];
          const action = c.status === 'treat'
            ? (t && t.treatment ? esc(t.treatment) : 'Treat this zone only.')
            : 'AI not sure here: check these plants or rescan before treating.';
          html += `<li><span class="z" style="background:${COLORS[c.status]}">${c.n}</span><span><b>${what}</b><small>${action}</small></span></li>`;
        }
        html += '</ul>';
      }
      html += `<div class="pdd-legend">
          <span><span class="pdd-dot" style="background:${COLORS.treat}"></span>Treat</span>
          <span><span class="pdd-dot" style="background:${COLORS.check}"></span>Not sure: check</span>
          <span><span class="pdd-dot" style="background:${COLORS.healthy}"></span>Healthy</span>
        </div>`;
      html += `<p class="pdd-note">Field split into ${grid.rows} × ${grid.cols} zones of ${grid.cell} m. ${scanned.length} zone${scanned.length === 1 ? '' : 's'} scanned so far.${noLoc ? ` ${noLoc} scan${noLoc > 1 ? 's have' : ' has'} no GPS location and ${noLoc > 1 ? 'are' : 'is'} not on the map.` : ''}${navigator.onLine ? '' : ' Offline: showing map tiles saved earlier.'}</p>`;
    }
    html += '<div class="pdd-btns">';
    if (grid) html += '<button type="button" data-act="fit">Zoom to field</button>';
    html += demoCount
      ? `<button type="button" data-act="demo-off">Remove ${demoCount} demo scans</button>`
      : '<button type="button" class="primary" data-act="demo-on">Load demo field (20 scans)</button>';
    html += '</div></div>';
    sumEl.innerHTML = html;
  }

  // ---------- demo data: 20 fake scans in one 120 m x 80 m field ----------
  function demoScans() {
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const mLat = 111320, mLng = 111320 * Math.cos((DEMO_CENTER[0] * Math.PI) / 180);
    const at = (xM, yM) => [DEMO_CENTER[0] + yM / mLat, DEMO_CENTER[1] + xM / mLng]; // x east, y north from centre
    const plan = [];
    // early blight hotspot in the north-east corner
    for (let i = 0; i < 5; i++) plan.push(['Tomato___Early_blight', 0.9 + rnd() * 0.09, 35 + rnd() * 20, 18 + rnd() * 18]);
    // septoria patch on the west edge
    for (let i = 0; i < 2; i++) plan.push(['Tomato___Septoria_leaf_spot', 0.86 + rnd() * 0.1, -52 + rnd() * 10, -8 + rnd() * 14]);
    // two scans the AI was not sure about
    plan.push(['Tomato___Target_Spot', 0.62, -5, 28]);
    plan.push(['Tomato___Bacterial_spot', 0.55, 20, -30]);
    // the rest healthy, spread across the field
    while (plan.length < 20) plan.push(['Tomato___healthy', 0.93 + rnd() * 0.069, -55 + rnd() * 110, -36 + rnd() * 72]);
    const now = Date.now();
    return plan.map(([label, conf, x, y], i) => {
      const [lat, lng] = at(x, y);
      return {
        id: 'demo-' + (i + 1), time: new Date(now - (20 - i) * 4 * 60000).toISOString(),
        lat, lng, label, confidence: Number(conf.toFixed(4)),
        top3: [{ label, confidence: Number(conf.toFixed(4)) }], thumb: null, accuracy: 5, gps: 'demo', demo: true,
      };
    });
  }

  // ---------- events ----------
  if (sumEl) sumEl.addEventListener('click', async (e) => {
    const b = e.target.closest('button[data-act]');
    if (!b) return;
    if (b.dataset.act === 'fit' && lastGrid) map.fitBounds(lastGrid.bounds, { padding: [20, 20], maxZoom: 19 });
    if (b.dataset.act === 'demo-on') { await window.PDDStorage.putMany(demoScans()); window.dispatchEvent(new CustomEvent('pdd:scans-changed')); }
    if (b.dataset.act === 'demo-off') await window.PDDStorage.clearScans({ demoOnly: true });
  });
  map.on('popupopen', (e) => {
    const btn = e.popup.getElement() && e.popup.getElement().querySelector('button[data-del]');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      if (btn.dataset.confirm) { map.closePopup(); await window.PDDStorage.deleteScan(btn.dataset.del); }
      else { btn.dataset.confirm = '1'; btn.textContent = 'Tap again to delete'; }
    });
  });
  window.addEventListener('pdd:scans-changed', render);
  window.addEventListener('pdd:scan-saved', render);
  window.addEventListener('pdd:show-map', () => { setTimeout(() => { map.invalidateSize(); if (lastGrid) fitField(lastGrid); }, 60); });
  window.addEventListener('online', render);
  window.addEventListener('offline', render);

  window.PDDMap = { map, render, buildGrid, demoScans, statusOf };
  render();
})();
