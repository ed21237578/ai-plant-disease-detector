/* PDD storage (Person 3).
 * Saves scans on the phone in IndexedDB and adds the GPS position.
 *
 * Globals:
 *   saveScan(scan)  -> Promise<savedScan>  adds lat/lng/accuracy (from GPS) and stores it
 *   getScans()      -> Promise<scan[]>     oldest first
 *   PDDStorage.deleteScan(id), PDDStorage.clearScans({ demoOnly }), PDDStorage.putMany(scans)
 * Events (on window):
 *   'pdd:scans-changed' after anything is saved or deleted
 *
 * Scan record (agreed format): { id, time, lat, lng, label, confidence, top3, thumb }
 * Extra fields added here: accuracy (metres, from GPS), gps ('ok' | 'denied' | 'unavailable' | 'timeout' | 'given'), demo (true for fake scans)
 */
(function () {
  'use strict';

  const DB_NAME = 'pdd';
  const STORE = 'scans';
  const GPS_TIMEOUT_MS = 12000;

  // ---------- IndexedDB (with an in-memory fallback, e.g. some private-browsing modes) ----------
  let dbPromise = null;
  const memory = new Map();
  let useMemory = !('indexedDB' in window);

  function openDb() {
    if (useMemory) return Promise.resolve(null);
    if (!dbPromise) {
      dbPromise = new Promise((resolve) => {
        let req;
        try { req = indexedDB.open(DB_NAME, 1); } catch (e) { useMemory = true; resolve(null); return; }
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains(STORE)) {
            const s = db.createObjectStore(STORE, { keyPath: 'id' });
            s.createIndex('time', 'time');
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => { useMemory = true; resolve(null); };
        req.onblocked = () => { useMemory = true; resolve(null); };
      });
    }
    return dbPromise;
  }

  function tx(db, mode, fn) {
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORE, mode);
      const store = t.objectStore(STORE);
      let result;
      Promise.resolve(fn(store)).then((r) => { result = r; });
      t.oncomplete = () => resolve(result);
      t.onerror = () => reject(t.error || new Error('Storage error'));
      t.onabort = () => reject(t.error || new Error('Storage aborted (phone storage may be full)'));
    });
  }

  const reqP = (req) => new Promise((res, rej) => { req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error); });

  async function putMany(scans) {
    const db = await openDb();
    if (!db) { scans.forEach((s) => memory.set(s.id, s)); return scans; }
    await tx(db, 'readwrite', (store) => { scans.forEach((s) => store.put(s)); });
    return scans;
  }

  async function getScans() {
    const db = await openDb();
    let all;
    if (!db) all = [...memory.values()];
    else all = await tx(db, 'readonly', (store) => reqP(store.getAll()));
    return (all || []).sort((a, b) => String(a.time).localeCompare(String(b.time)));
  }

  async function deleteScan(id) {
    const db = await openDb();
    if (!db) memory.delete(id);
    else await tx(db, 'readwrite', (store) => { store.delete(id); });
    changed();
  }

  async function clearScans(opts) {
    const demoOnly = !!(opts && opts.demoOnly);
    const all = await getScans();
    const ids = all.filter((s) => !demoOnly || s.demo).map((s) => s.id);
    const db = await openDb();
    if (!db) ids.forEach((id) => memory.delete(id));
    else await tx(db, 'readwrite', (store) => { ids.forEach((id) => store.delete(id)); });
    changed();
    return ids.length;
  }

  function changed() { window.dispatchEvent(new CustomEvent('pdd:scans-changed')); }

  // ---------- GPS ----------
  function getPosition() {
    return new Promise((resolve) => {
      if (!('geolocation' in navigator)) { resolve({ gps: 'unavailable' }); return; }
      navigator.geolocation.getCurrentPosition(
        (pos) => resolve({ gps: 'ok', lat: pos.coords.latitude, lng: pos.coords.longitude, accuracy: Math.round(pos.coords.accuracy) }),
        (err) => resolve({ gps: err.code === 1 ? 'denied' : err.code === 3 ? 'timeout' : 'unavailable' }),
        { enableHighAccuracy: true, timeout: GPS_TIMEOUT_MS, maximumAge: 15000 }
      );
    });
  }

  // ---------- public API ----------
  async function saveScan(scan) {
    if (!scan || !scan.label) throw new Error('Scan is missing its result');
    const rec = Object.assign({}, scan);
    if (!rec.id) rec.id = (crypto.randomUUID ? crypto.randomUUID() : 'scan-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8));
    if (!rec.time) rec.time = new Date().toISOString();

    if (typeof rec.lat === 'number' && typeof rec.lng === 'number') {
      rec.gps = rec.gps || 'given';
    } else {
      const pos = await getPosition();
      rec.gps = pos.gps;
      rec.lat = pos.gps === 'ok' ? pos.lat : null;
      rec.lng = pos.gps === 'ok' ? pos.lng : null;
      if (pos.gps === 'ok') rec.accuracy = pos.accuracy;
    }
    // The scan is always kept, even without a location, so a farmer never loses a result.
    await putMany([rec]);
    changed();
    return rec;
  }

  window.saveScan = saveScan;
  window.getScans = getScans;
  window.PDDStorage = { saveScan, getScans, deleteScan, clearScans, putMany, getPosition };
})();
