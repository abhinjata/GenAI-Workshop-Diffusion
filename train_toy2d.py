"""
Bonus - 2D toy diffusion: an MLP learns to turn a Gaussian blob into the word ENIGMA.
Trains in ~5 min on CPU and exports weights for the browser lab (station 4).

    python train_toy2d.py
"""
import os, numpy as np, torch, torch.nn as nn, json, math
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)
# Rasterize "ENIGMA" and sample points inside the glyphs
fig = plt.figure(figsize=(8, 2), dpi=100); fig.text(0.5, 0.5, 'ENIGMA', ha='center', va='center', fontsize=120, weight='bold', family='DejaVu Sans')
fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[:, :, 0] < 128; plt.close(fig)
ys, xs = np.nonzero(img)
pts = np.stack([xs, -ys], 1).astype(np.float32)
pts -= pts.mean(0); pts /= pts[:, 0].std() / 1.6
print('pts', len(pts), pts.min(0), pts.max(0))
data = torch.tensor(pts)
T = 1000; betas = torch.linspace(1e-4, 0.02, T); ab = torch.cumprod(1 - betas, 0)
def temb(t, d=32):
    half = d // 2; f = torch.exp(-math.log(10000) * torch.arange(half) / half)
    a = t[:, None].float() * f[None]; return torch.cat([torch.cos(a), torch.sin(a)], 1)
class MLP(nn.Module):
    def __init__(s, h=256):
        super().__init__(); s.l1 = nn.Linear(2 + 2 * 2 * 6 + 32, h); s.l2 = nn.Linear(h, h); s.l3 = nn.Linear(h, h); s.l4 = nn.Linear(h, 2); s.a = nn.SiLU()
    def forward(s, x, t):
        fr = x[:, :, None] * (2.0 ** torch.arange(6))[None, None] ; ff = torch.cat([torch.sin(fr), torch.cos(fr)], 1).flatten(1)
        h = s.a(s.l1(torch.cat([x, ff, temb(t)], 1))); h = s.a(s.l2(h)) + h; h = s.a(s.l3(h)) + h; return s.l4(h)
m = MLP(); opt = torch.optim.Adam(m.parameters(), 2e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20000)
for it in range(20000):
    x0 = data[torch.randint(0, len(data), (1024,))]; t = torch.randint(0, T, (1024,)); e = torch.randn_like(x0)
    xt = ab[t, None].sqrt() * x0 + (1 - ab[t, None]).sqrt() * e
    loss = ((m(xt, t) - e) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if it % 3000 == 0: print(it, loss.item(), flush=True)
# sample check
with torch.no_grad():
    x = torch.randn(3000, 2); steps = list(range(999, -1, -10))
    for i, t in enumerate(steps):
        tp = steps[i + 1] if i + 1 < len(steps) else -1
        a_t = ab[t]; a_p = ab[tp] if tp >= 0 else torch.tensor(1.0)
        eps = m(x, torch.full((3000,), t)); x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        cb = 1 - a_t / a_p; mean = (a_p.sqrt() * cb / (1 - a_t)) * x0 + ((a_t / a_p).sqrt() * (1 - a_p) / (1 - a_t)) * x
        var = (1 - a_p) / (1 - a_t) * cb
        x = mean + (var.sqrt() * torch.randn_like(x) if tp >= 0 else 0)
plt.figure(figsize=(8, 3)); plt.scatter(x[:, 0], x[:, 1], s=1); plt.axis('equal'); plt.title('3000 samples'); plt.show()
# Export: weights as float32 (linear layers transposed to [in, out]) + 1500 data points.
man, arr, off = {}, [], 0
for k, v in m.state_dict().items():
    a = (v.T if v.ndim == 2 else v).contiguous().numpy().astype(np.float32)
    man[k] = [off, list(a.shape)]; arr.append(a.ravel()); off += a.size
np.concatenate(arr).tofile(os.path.join(WEB, 'toy2d.bin'))
import base64; open(os.path.join(WEB, 'toy2d.bin.txt'), 'w').write(base64.b64encode(np.concatenate(arr).tobytes()).decode())
sub = pts[np.random.choice(len(pts), 1500, replace=False)]
json.dump({'manifest': man, 'data': np.round(sub, 3).tolist()}, open(os.path.join(WEB, 'toy2d.json'), 'w'))
print('exported to', WEB)
