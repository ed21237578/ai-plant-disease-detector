import base64, json, sys
import numpy as np
import cnn

src = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/demo/model.npz'
dst = sys.argv[2] if len(sys.argv) > 2 else '/home/claude/demo/model.json'
m = np.load(src)
meta = json.load(open('/home/claude/demo/meta.json'))
parts = []
for i in range(len(cnn.CONVS)):
    parts += [m[f'W{i}'], m[f'b{i}']]
parts += [m['Wfc'], m['bfc']]
blob = np.concatenate([p.astype('<f4').reshape(-1) for p in parts]).tobytes()
out = {
    'classes': meta['classes'],
    'mean': [float(v) for v in m['mean']],
    'std': [float(v) for v in m['std']],
    'convs': [dict(cin=ci, cout=co, stride=s, pool=pool) for ci, co, s, pool in cnn.CONVS],
    'weights': base64.b64encode(blob).decode(),
}
json.dump(out, open(dst, 'w'))
print('exported', dst, len(blob), 'bytes of weights')
