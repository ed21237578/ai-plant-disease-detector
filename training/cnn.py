"""Small CNN in pure NumPy (NHWC, float32) for leaf disease classification.

Architecture (input 64x64x3):
  conv1 3x3 s2 3->16  + ReLU            -> 32x32x16
  conv2 3x3 s1 16->32 + ReLU + maxpool2 -> 16x16x32
  conv3 3x3 s1 32->64 + ReLU + maxpool2 -> 8x8x64
  conv4 3x3 s1 64->128 + ReLU           -> 8x8x128
  global average pool                   -> 128
  fc 128->K                             -> logits
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

CONVS = [(3, 16, 2, False), (16, 32, 1, True), (32, 64, 1, True), (64, 128, 1, False)]  # cin, cout, stride, pool


def init_params(K, rng, dtype=np.float32):
    p = {}
    for i, (ci, co, s, pool) in enumerate(CONVS):
        fan_in = 9 * ci
        p[f'W{i}'] = (rng.standard_normal((9 * ci, co)) * np.sqrt(2.0 / fan_in)).astype(dtype)
        p[f'b{i}'] = np.zeros(co, dtype)
    p['Wfc'] = (rng.standard_normal((CONVS[-1][1], K)) * np.sqrt(1.0 / CONVS[-1][1])).astype(dtype)
    p['bfc'] = np.zeros(K, dtype)
    return p


def im2col(x, stride):
    # x: (N,H,W,C) -> cols (N*Ho*Wo, 9*C) ordered (kh, kw, c); pad=1
    N, H, W, C = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    win = sliding_window_view(xp, (3, 3), axis=(1, 2))  # (N,H,W,C,3,3)
    win = win[:, ::stride, ::stride]
    Ho, Wo = win.shape[1], win.shape[2]
    cols = np.ascontiguousarray(win.transpose(0, 1, 2, 4, 5, 3)).reshape(N * Ho * Wo, 9 * C)
    return cols, Ho, Wo


def col2im(dcols, xshape, stride, Ho, Wo):
    N, H, W, C = xshape
    d = dcols.reshape(N, Ho, Wo, 3, 3, C)
    dxp = np.zeros((N, H + 2, W + 2, C), dcols.dtype)
    for i in range(3):
        for j in range(3):
            dxp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride, :] += d[:, :, :, i, j, :]
    return dxp[:, 1:-1, 1:-1, :]


def maxpool_fwd(x):
    N, H, W, C = x.shape
    r = x.reshape(N, H // 2, 2, W // 2, 2, C).transpose(0, 1, 3, 2, 4, 5).reshape(N, H // 2, W // 2, 4, C)
    idx = r.argmax(axis=3)
    out = np.take_along_axis(r, idx[:, :, :, None, :], axis=3)[:, :, :, 0, :]
    return out, idx


def maxpool_bwd(dout, idx, xshape):
    N, H, W, C = xshape
    dr = np.zeros((N, H // 2, W // 2, 4, C), dout.dtype)
    np.put_along_axis(dr, idx[:, :, :, None, :], dout[:, :, :, None, :], axis=3)
    return dr.reshape(N, H // 2, W // 2, 2, 2, C).transpose(0, 1, 3, 2, 4, 5).reshape(N, H, W, C)


def forward(p, x, train=False):
    caches = []
    h = x
    for i, (ci, co, s, pool) in enumerate(CONVS):
        cols, Ho, Wo = im2col(h, s)
        z = (cols @ p[f'W{i}'] + p[f'b{i}']).reshape(h.shape[0], Ho, Wo, co)
        a = np.maximum(z, 0)
        cache = dict(xshape=h.shape, cols=cols if train else None, Ho=Ho, Wo=Wo, a=a if train else None)
        if pool:
            a2, idx = maxpool_fwd(a)
            cache['idx'] = idx
            cache['ashape'] = a.shape
            a = a2
        caches.append(cache)
        h = a
    feat = h  # (N,8,8,128)
    g = feat.mean(axis=(1, 2))
    logits = g @ p['Wfc'] + p['bfc']
    return logits, (caches, feat, g)


def loss_and_grads(p, x, y, wd=0.0):
    logits, (caches, feat, g) = forward(p, x, train=True)
    N = x.shape[0]
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    prob = e / e.sum(axis=1, keepdims=True)
    loss = -np.log(prob[np.arange(N), y] + 1e-12).mean()
    dl = prob.copy()
    dl[np.arange(N), y] -= 1
    dl /= N
    grads = {'Wfc': g.T @ dl, 'bfc': dl.sum(0)}
    dg = dl @ p['Wfc'].T
    Hf, Wf = feat.shape[1], feat.shape[2]
    dh = np.broadcast_to(dg[:, None, None, :] / (Hf * Wf), feat.shape).astype(x.dtype)
    for i in reversed(range(len(CONVS))):
        ci, co, s, pool = CONVS[i]
        c = caches[i]
        if pool:
            dh = maxpool_bwd(dh, c['idx'], c['ashape'])
        dz = dh * (c['a'] > 0)
        dz2 = dz.reshape(-1, co)
        grads[f'W{i}'] = c['cols'].T @ dz2
        grads[f'b{i}'] = dz2.sum(0)
        if i > 0:
            dcols = dz2 @ p[f'W{i}'].T
            dh = col2im(dcols, c['xshape'], s, c['Ho'], c['Wo'])
    if wd:
        for k in p:
            if k.startswith('W'):
                loss += 0.5 * wd * float((p[k] ** 2).sum())
                grads[k] = grads[k] + wd * p[k]
    return loss, grads, logits


class Adam:
    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.t = 0

    def step(self, p, g):
        self.t += 1
        for k in p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] * g[k]
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p[k] -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(p[k].dtype)
