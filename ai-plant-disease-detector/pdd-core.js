// PDD core: preprocessing + CNN inference (exact port of cnn.py / prep.py). No dependencies.
(function (root) {
  'use strict';

  function b64ToF32(b64) {
    let bin;
    if (typeof atob === 'function') bin = atob(b64);
    else bin = Buffer.from(b64, 'base64').toString('binary');
    const buf = new ArrayBuffer(bin.length);
    const u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(buf);
  }

  // model: {classes, mean, std, convs:[{cin,cout,stride,pool}], weights: base64 of all float32 in order W0,b0,...,W3,b3,Wfc,bfc}
  function loadModel(m) {
    const all = b64ToF32(m.weights);
    let off = 0;
    const take = (n) => { const a = all.subarray(off, off + n); off += n; return a; };
    const layers = m.convs.map((c) => ({ ...c, W: take(9 * c.cin * c.cout), b: take(c.cout) }));
    const F = m.convs[m.convs.length - 1].cout, K = m.classes.length;
    const Wfc = take(F * K), bfc = take(K);
    if (off !== all.length) throw new Error('weight size mismatch');
    return { classes: m.classes, mean: m.mean, std: m.std, layers, Wfc, bfc, F, K };
  }

  // x: Float32Array NHWC (H,W,C) single image; pad=1, 3x3 kernel
  function conv3x3(x, H, Wd, C, L) {
    const s = L.stride, Ho = Math.floor((H - 1) / s) + 1, Wo = Math.floor((Wd - 1) / s) + 1, Co = L.cout;
    const out = new Float32Array(Ho * Wo * Co);
    const W = L.W, b = L.b;
    const acc = new Float32Array(Co);
    for (let oy = 0; oy < Ho; oy++) {
      for (let ox = 0; ox < Wo; ox++) {
        acc.set(b);
        for (let ky = 0; ky < 3; ky++) {
          const iy = oy * s + ky - 1;
          if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < 3; kx++) {
            const ix = ox * s + kx - 1;
            if (ix < 0 || ix >= Wd) continue;
            const inBase = (iy * Wd + ix) * C;
            const wBase = (ky * 3 + kx) * C;
            for (let c = 0; c < C; c++) {
              const v = x[inBase + c];
              if (v === 0) continue;
              const wr = (wBase + c) * Co;
              for (let o = 0; o < Co; o++) acc[o] += v * W[wr + o];
            }
          }
        }
        const ob = (oy * Wo + ox) * Co;
        for (let o = 0; o < Co; o++) out[ob + o] = acc[o] > 0 ? acc[o] : 0; // ReLU
      }
    }
    return { y: out, H: Ho, W: Wo, C: Co };
  }

  function maxpool2(t) {
    const Ho = t.H / 2, Wo = t.W / 2, C = t.C, out = new Float32Array(Ho * Wo * C);
    for (let y = 0; y < Ho; y++) for (let x = 0; x < Wo; x++) for (let c = 0; c < C; c++) {
      const i00 = ((2 * y) * t.W + 2 * x) * C + c, i01 = i00 + C, i10 = i00 + t.W * C, i11 = i10 + C;
      out[(y * Wo + x) * C + c] = Math.max(t.y[i00], t.y[i01], t.y[i10], t.y[i11]);
    }
    return { y: out, H: Ho, W: Wo, C };
  }

  // rgb64: Float32Array 64*64*3 of raw 0..255 values (already box-averaged)
  function forward(model, rgb64) {
    const x = new Float32Array(rgb64.length);
    for (let i = 0; i < rgb64.length; i += 3) for (let c = 0; c < 3; c++) x[i + c] = (rgb64[i + c] - model.mean[c]) / model.std[c];
    let t = { y: x, H: 64, W: 64, C: 3 };
    for (const L of model.layers) {
      t = conv3x3(t.y, t.H, t.W, t.C, L);
      if (L.pool) t = maxpool2(t);
    }
    const F = t.C, HW = t.H * t.W, g = new Float32Array(F);
    for (let i = 0; i < HW; i++) for (let f = 0; f < F; f++) g[f] += t.y[i * F + f];
    for (let f = 0; f < F; f++) g[f] /= HW;
    const K = model.K, logits = new Float32Array(K);
    for (let k = 0; k < K; k++) {
      let s = model.bfc[k];
      for (let f = 0; f < F; f++) s += g[f] * model.Wfc[f * K + k];
      logits[k] = s;
    }
    return { logits, feat: t };
  }

  function softmax(l) {
    let m = -Infinity; for (const v of l) m = Math.max(m, v);
    const e = Array.from(l, (v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  // Class activation map for class k: sum_f Wfc[f,k] * feat[h,w,f]  -> Float32Array(H*W), normalised 0..1
  function cam(model, feat, k) {
    const F = feat.C, HW = feat.H * feat.W, out = new Float32Array(HW);
    for (let i = 0; i < HW; i++) { let s = 0; for (let f = 0; f < F; f++) s += model.Wfc[f * model.K + k] * feat.y[i * F + f]; out[i] = s; }
    let lo = Infinity, hi = -Infinity; for (const v of out) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (let i = 0; i < HW; i++) out[i] = hi > lo ? (out[i] - lo) / (hi - lo) : 0;
    return { map: out, H: feat.H, W: feat.W };
  }

  // rgba256: Uint8ClampedArray 256*256*4 -> Float32Array 64*64*3 (4x4 box average, same as prep.py)
  function boxDown(rgba256) {
    const out = new Float32Array(64 * 64 * 3);
    for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
      let r = 0, g = 0, b = 0;
      for (let dy = 0; dy < 4; dy++) for (let dx = 0; dx < 4; dx++) {
        const i = ((4 * y + dy) * 256 + (4 * x + dx)) * 4;
        r += rgba256[i]; g += rgba256[i + 1]; b += rgba256[i + 2];
      }
      const o = (y * 64 + x) * 3; out[o] = r / 16; out[o + 1] = g / 16; out[o + 2] = b / 16;
    }
    return out;
  }

  // ---- image checks on the 256x256 frame ----
  function rgb2hsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let h = 0;
    if (d > 0) {
      if (mx === r) h = 60 * (((g - b) / d) % 6);
      else if (mx === g) h = 60 * ((b - r) / d + 2);
      else h = 60 * ((r - g) / d + 4);
    }
    if (h < 0) h += 360;
    return [h, mx === 0 ? 0 : d / mx, mx];
  }

  // leaf = saturated plant-coloured pixels; lesion = leaf pixels that are yellow/brown rather than green
  function leafStats(rgba) {
    let leaf = 0, lesion = 0; const n = rgba.length / 4;
    for (let i = 0; i < rgba.length; i += 4) {
      const [h, s, v] = rgb2hsv(rgba[i], rgba[i + 1], rgba[i + 2]);
      if (s > 0.25 && v > 0.15 && h >= 15 && h <= 170) {
        leaf++;
        if (h < 65) lesion++;
      }
    }
    return { leafFrac: leaf / n, lesionFrac: leaf ? lesion / leaf : 0 };
  }

  // variance of the 4-neighbour Laplacian on grayscale (sharpness)
  function sharpness(rgba, size) {
    const gray = new Float32Array(size * size);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j++) gray[j] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    let s = 0, s2 = 0, n = 0;
    for (let y = 1; y < size - 1; y++) for (let x = 1; x < size - 1; x++) {
      const i = y * size + x;
      const l = gray[i - 1] + gray[i + 1] + gray[i - size] + gray[i + size] - 4 * gray[i];
      s += l; s2 += l * l; n++;
    }
    const m = s / n; return s2 / n - m * m;
  }

  const api = { loadModel, forward, softmax, cam, boxDown, leafStats, sharpness };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PDD = api;
})(typeof window !== 'undefined' ? window : globalThis);
