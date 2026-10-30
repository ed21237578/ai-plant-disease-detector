/* PDD service worker (Person 3): makes the app open and work with no signal.
 * - App files + AI model: cached on first visit, served from cache, refreshed in the background.
 * - Map tiles: cached as you view them, so fields you have looked at still show offline.
 * After changing any app file, bump VERSION so phones pick up the new files.
 */
const VERSION = 'v2';
const APP_CACHE = 'pdd-app-' + VERSION;
const TILE_CACHE = 'pdd-tiles-v1';
const MAX_TILES = 600;
const TILE_HOSTS = ['server.arcgisonline.com', 'tile.openstreetmap.org'];

const APP_FILES = [
  './', 'index.html', 'style.css', 'scan.js', 'pdd-core.js', 'model.json', 'treatments.json',
  'storage.js', 'map.js', 'manifest.json', 'model.js', 'treatments.js', 'app.js',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/maskable-512.png', 'icons/apple-touch-icon.png',
  'vendor/leaflet/leaflet.js', 'vendor/leaflet/leaflet.css',
  'vendor/leaflet/images/layers.png', 'vendor/leaflet/images/layers-2x.png',
  'vendor/leaflet/images/marker-icon.png', 'vendor/leaflet/images/marker-icon-2x.png', 'vendor/leaflet/images/marker-shadow.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(APP_CACHE);
    // add one by one so a single missing file never breaks offline mode
    await Promise.all(APP_FILES.map((url) => cache.add(new Request(url, { cache: 'reload' })).catch(() => {})));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keep = [APP_CACHE, TILE_CACHE];
    for (const key of await caches.keys()) if (!keep.includes(key)) await caches.delete(key);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (TILE_HOSTS.includes(url.hostname)) { event.respondWith(tile(req)); return; }
  if (url.origin !== self.location.origin) return;
  if (req.mode === 'navigate') { event.respondWith(page(req)); return; }
  event.respondWith(staleWhileRevalidate(req));
});

// Pages: try the network first (fresh after a deploy), fall back to the cached app when offline.
async function page(req) {
  const cache = await caches.open(APP_CACHE);
  try {
    const res = await fetch(req);
    if (res.ok) cache.put('index.html', res.clone());
    return res;
  } catch (e) {
    return (await cache.match(req, { ignoreSearch: true })) || (await cache.match('index.html')) || (await cache.match('./')) || Response.error();
  }
}

// App files: answer from cache straight away, update the cache in the background.
async function staleWhileRevalidate(req) {
  const cache = await caches.open(APP_CACHE);
  const cached = await cache.match(req, { ignoreSearch: true });
  const network = fetch(req).then((res) => { if (res.ok) cache.put(req, res.clone()); return res; }).catch(() => null);
  if (cached) return cached;
  return (await network) || new Response('Offline and not cached yet', { status: 503, statusText: 'Offline' });
}

// Map tiles: cache-first. Fetched in CORS mode so they can be stored efficiently; if a server
// refuses CORS the tile still shows, it just isn't saved for offline use.
async function tile(req) {
  const cache = await caches.open(TILE_CACHE);
  const cached = await cache.match(req.url);
  if (cached) return cached;
  try {
    const res = await fetch(req.url, { mode: 'cors', credentials: 'omit' });
    if (res.ok) { await cache.put(req.url, res.clone()); trimTiles(cache); }
    return res;
  } catch (e) {
    try { return await fetch(req); } catch (e2) { return Response.error(); }
  }
}

let trimming = false;
async function trimTiles(cache) {
  if (trimming) return;
  trimming = true;
  try {
    const keys = await cache.keys();
    for (let i = 0; i < keys.length - MAX_TILES; i++) await cache.delete(keys[i]);
  } finally { trimming = false; }
}
