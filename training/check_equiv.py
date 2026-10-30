"""Check the JS inference matches NumPy exactly on held-out test images."""
import json, subprocess, sys
import numpy as np
import cnn

model_npz = sys.argv[1]
model_json = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 937
m = np.load(model_npz)
p = {k: m[k] for k in m.files if k not in ('mean', 'std')}
d = np.load('/home/claude/demo/data.npz')
X = d['X_test'][:n]
y = d['y_test'][:n]
Xn = ((X - m['mean']) / m['std']).astype(np.float32)
ref = np.concatenate([cnn.forward(p, Xn[i:i + 128])[0] for i in range(0, n, 128)])
X.astype('<f4').tofile('/tmp/xtest.bin')
js = subprocess.run(['node', '-e', f'''
const fs=require('fs'); const P=require('/home/claude/demo/pdd-core.js');
const M=P.loadModel(JSON.parse(fs.readFileSync('{model_json}','utf8')));
const buf=fs.readFileSync('/tmp/xtest.bin'); const all=new Float32Array(buf.buffer, buf.byteOffset, buf.length/4);
const out=[]; const t=Date.now();
for(let i=0;i<{n};i++){{ out.push(Array.from(P.forward(M, all.subarray(i*12288,(i+1)*12288)).logits)); }}
console.error('ms/image', (Date.now()-t)/{n});
console.log(JSON.stringify(out));
'''], capture_output=True, text=True)
print(js.stderr.strip())
jl = np.array(json.loads(js.stdout))
diff = np.abs(jl - ref).max()
agree = (jl.argmax(1) == ref.argmax(1)).mean()
print(f'images {n}  max |logit diff| {diff:.2e}  argmax agreement {agree*100:.2f}%  js acc {(jl.argmax(1)==y).mean()*100:.2f}%')
assert diff < 1e-3 and agree == 1.0, 'JS/NumPy MISMATCH'
print('JS MATCHES NUMPY')
