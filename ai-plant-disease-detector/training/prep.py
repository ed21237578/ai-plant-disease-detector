"""Preprocess: centre-crop square -> 256x256 (bilinear) -> 4x4 box average -> 64x64 float32 RGB (0..255).
The browser demo replicates exactly this pipeline."""
import glob, json, os, random
import numpy as np
from PIL import Image

ROOT = '/home/claude/demo/pv/raw/color'
CLASSES = sorted(os.listdir(ROOT))


def load(path):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    if im.size != (256, 256):
        im = im.resize((256, 256), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    return a.reshape(64, 4, 64, 4, 3).mean(axis=(1, 3))


if __name__ == '__main__':
    splits = {'train': [], 'val': [], 'test': []}
    for ci, c in enumerate(CLASSES):
        fs = sorted(glob.glob(f'{ROOT}/{c}/*'))
        random.Random(7).shuffle(fs)
        n = len(fs)
        nte = round(n * 0.15); nva = round(n * 0.15)
        splits['test'] += [(f, ci) for f in fs[:nte]]
        splits['val'] += [(f, ci) for f in fs[nte:nte + nva]]
        splits['train'] += [(f, ci) for f in fs[nte + nva:]]
    out = {}
    for k, items in splits.items():
        X = np.stack([load(f) for f, _ in items]).astype(np.float32)
        y = np.array([c for _, c in items], np.int64)
        out[f'X_{k}'] = X; out[f'y_{k}'] = y
        print(k, X.shape, np.bincount(y, minlength=len(CLASSES)))
    np.savez('/home/claude/demo/data.npz', **out)
    json.dump({'classes': CLASSES, 'test_files': [f for f, _ in splits['test']]}, open('/home/claude/demo/meta.json', 'w'), indent=1)
