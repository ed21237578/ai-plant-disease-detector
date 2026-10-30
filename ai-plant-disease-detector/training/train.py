import json, sys, time
import numpy as np
import cnn

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BS = 64
LR = 2e-3
WD = 5e-4

d = np.load('/home/claude/demo/data.npz')
meta = json.load(open('/home/claude/demo/meta.json'))
K = len(meta['classes'])
Xtr, ytr, Xva, yva, Xte, yte = d['X_train'], d['y_train'], d['X_val'], d['y_val'], d['X_test'], d['y_test']
mean = Xtr.mean(axis=(0, 1, 2)).astype(np.float32)
std = Xtr.std(axis=(0, 1, 2)).astype(np.float32)
norm = lambda X: ((X - mean) / std).astype(np.float32)
Xva_n, Xte_n = norm(Xva), norm(Xte)
rng = np.random.default_rng(123)


def augment(X):
    X = X.copy()
    n = X.shape[0]
    # flips + rot90
    f = rng.random(n) < 0.5; X[f] = X[f, :, ::-1]
    f = rng.random(n) < 0.5; X[f] = X[f, ::-1, :]
    k = rng.integers(0, 4, n)
    for r in (1, 2, 3):
        m = k == r
        if m.any():
            X[m] = np.rot90(X[m], r, axes=(1, 2))
    # brightness / contrast jitter
    X = X * rng.uniform(0.8, 1.2, (n, 1, 1, 1)) + rng.uniform(-20, 20, (n, 1, 1, 1))
    X = np.clip(X, 0, 255)
    # random shift via reflect pad 6 + crop
    P = np.pad(X, ((0, 0), (6, 6), (6, 6), (0, 0)), mode='reflect')
    out = np.empty_like(X)
    ox, oy = rng.integers(0, 13, n), rng.integers(0, 13, n)
    for i in range(n):
        out[i] = P[i, oy[i]:oy[i] + 64, ox[i]:ox[i] + 64]
    return out


def predict(p, X, bs=256):
    return np.concatenate([cnn.forward(p, X[i:i + bs])[0] for i in range(0, len(X), bs)])


p = cnn.init_params(K, rng)
opt = cnn.Adam(p, lr=LR)
steps_per_epoch = int(np.ceil(len(Xtr) / BS))
total = EPOCHS * steps_per_epoch
best, best_p = -1, None
step = 0
for ep in range(EPOCHS):
    t0 = time.time()
    perm = rng.permutation(len(Xtr))
    losses = []
    for b in range(steps_per_epoch):
        idx = perm[b * BS:(b + 1) * BS]
        xb = norm(augment(Xtr[idx]))
        opt.lr = 1e-5 + 0.5 * (LR - 1e-5) * (1 + np.cos(np.pi * step / total))
        loss, g, _ = cnn.loss_and_grads(p, xb, ytr[idx], wd=WD)
        opt.step(p, g)
        losses.append(loss)
        step += 1
    va = (predict(p, Xva_n).argmax(1) == yva).mean()
    if va > best:
        best, best_p = va, {k: v.copy() for k, v in p.items()}
    print(f'epoch {ep+1}/{EPOCHS} loss {np.mean(losses):.3f} val_acc {va:.4f} lr {opt.lr:.5f} time {time.time()-t0:.0f}s', flush=True)

p = best_p
logits = predict(p, Xte_n)
pred = logits.argmax(1)
te = (pred == yte).mean()
print(f'BEST VAL {best:.4f}  TEST ACC {te:.4f}  (n_test={len(yte)})')
cm = np.zeros((K, K), int)
for a, b in zip(yte, pred):
    cm[a, b] += 1
for i, c in enumerate(meta['classes']):
    print(f'{c:50s} recall {cm[i,i]/cm[i].sum():.3f}  n={cm[i].sum()}')
np.savez('/home/claude/demo/model.npz', mean=mean, std=std, **p)
np.save('/home/claude/demo/test_logits.npy', logits)
