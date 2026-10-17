# Synthesises trailer 3's sound effects and mixes them under the music:
# public/trailer3.wav (the spliced track) -> public/trailer3-mix.wav.
# Timings come from src/beats3.json, src/cut3.json and each recording's
# events.json (its real clicks and keystrokes), so sounds land on the action.
#
#   python scripts/sfx3.py      (numpy and soundfile)

import json
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path(__file__).resolve().parent.parent
music, SR = sf.read(root / "public/trailer3.wav", always_2d=True)
if music.shape[1] == 1:
    music = np.repeat(music, 2, axis=1)
beats = json.loads((root / "src/beats3.json").read_text())
cut = json.loads((root / "src/cut3.json").read_text())["clips"]
B = beats["bars"]
rng = np.random.default_rng(7)
fx = np.zeros_like(music)


def tt(d):
    return np.arange(int(d * SR)) / SR


def env(n, attack=0.004, decay=0.1):
    t = np.arange(n) / SR
    return np.minimum(1, t / attack) * np.exp(-t / decay)


def add(at, x, gain=1.0, pan=0.0):
    i = int(at * SR)
    if i < 0 or i >= len(fx):
        return
    x = x[: len(fx) - i] * gain / max(1e-9, np.abs(x).max())
    fx[i : i + len(x), 0] += x * (1 - max(0, pan))
    fx[i : i + len(x), 1] += x * (1 + min(0, pan))


def lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = (1 - a) * v + a * acc
        y[i] = acc
    return y


def tick(gain=1.0):
    n = int(0.018 * SR)
    x = rng.standard_normal(n) * env(n, 0.0005, 0.003)
    return (x - lowpass(x, 2500)) * gain


def key():
    n = int(0.03 * SR)
    x = rng.standard_normal(n) * env(n, 0.0005, 0.006)
    body = np.sin(2 * np.pi * rng.uniform(1700, 2300) * tt(0.03)) * env(n, 0.0005, 0.008)
    return (x - lowpass(x, 1800)) * 0.6 + body * 0.25


def chime(notes=(1318.5, 1760, 2093, 2637), step=0.045, decay=0.5):
    out = np.zeros(int((step * len(notes) + decay * 4) * SR))
    for k, hz in enumerate(notes):
        d = decay * 4
        tone = np.sin(2 * np.pi * hz * tt(d)) + 0.3 * np.sin(2 * np.pi * hz * 2.01 * tt(d))
        i = int(k * step * SR)
        out[i : i + len(tone)] += tone * env(len(tone), 0.002, decay)
    return out * 0.25


def whoosh(d=0.7, f0=4000, f1=300, up=False):
    n = int(d * SR)
    noise = rng.standard_normal(n)
    # A swept band: lowpass at a moving cutoff, minus a lower one.
    cut_hi = np.geomspace(f0, f1, n) if not up else np.geomspace(f1, f0, n)
    y = np.empty(n)
    lo = np.empty(n)
    a1 = a2 = 0.0
    for i in range(n):
        k = np.exp(-2 * np.pi * cut_hi[i] / SR)
        a1 = (1 - k) * noise[i] + k * a1
        k2 = np.exp(-2 * np.pi * cut_hi[i] * 0.35 / SR)
        a2 = (1 - k2) * noise[i] + k2 * a2
        y[i] = a1
        lo[i] = a2
    shape = np.sin(np.pi * np.linspace(0, 1, n) ** (0.6 if not up else 1.6))
    return (y - lo) * shape * 0.9


def boom(hz=52, d=1.2):
    t = tt(d)
    f = hz * (1 + 1.5 * np.exp(-t / 0.03))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.002, 0.35)
    return x


def boop(hz=660, down=True, d=0.14):
    t = tt(d)
    f = hz * (1 - 0.45 * t / d) if down else hz * (1 + 0.6 * t / d)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.002, 0.05) * 0.5


def thud(d=0.9):
    t = tt(d)
    f = 70 * (1 + 2 * np.exp(-t / 0.02))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.001, 0.18)
    n = rng.standard_normal(len(t)) * env(len(t), 0.0005, 0.03)
    slap = lowpass(n, 1800)
    return body + slap * 3


def slide(f0=1500, f1=260, d=0.65):
    t = tt(d)
    f = np.geomspace(f0, f1, len(t)) * (1 + 0.02 * np.sin(2 * np.pi * 7 * t))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return x * np.minimum(1, t / 0.02) * np.minimum(1, (d - t) / 0.08) * 0.32


def boing(d=0.5):
    t = tt(d)
    f = 180 + 260 * (1 - np.exp(-t / 0.08)) + 40 * np.sin(2 * np.pi * 14 * t) * np.exp(-t / 0.2)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.003, 0.18) * 0.5


def remap(m, t):
    xs = [p[0] for p in m]
    ys = [p[1] for p in m]
    # Clip time -> trailer time: invert the piecewise-linear map.
    return float(np.interp(t, ys, xs))


def clip_events(name, kinds):
    c = cut[name]
    events = json.loads((root / f"public/footage3/{name}/events.json").read_text())
    for e in events:
        if e["kind"] in kinds and c["map"][0][1] <= e["t"] <= c["map"][-1][1]:
            yield c["from"] + remap(c["map"], e["t"])


# 1. The opening: bits tick in, the pass hits, powers of ten.
for i in range(16):
    add(0.3 + i * 0.07, tick(), 1.6, pan=-0.4 + 0.05 * i)
add(B[1] - 0.45, whoosh(0.5, 300, 3000, up=True), 0.25)
add(B[1], chime(), 0.85)
# One long riser under the pull-back; a blip as each level fills the view.
dist = json.loads((root / "src/opening3.json").read_text())["dist"]
reached = [next(i for i, d in enumerate(dist) if d >= lv) / 30 for lv in (70, 1500, 8500, 30000)]
z0 = B[1] + 0.75
add(z0, whoosh(reached[-1] - z0 + 0.4, 2500, 150, up=True), 0.55)
for k, t in enumerate(reached[:-1]):
    add(t, boop([988, 1175, 1397][k], down=False, d=0.08), 0.5)
add(reached[-1], boom(), 0.75)

# 2. Typing the question; "Mary" appears.
for t in clip_events("hook", {"key"}):
    add(t, key(), 1.0, pan=rng.uniform(-0.2, 0.2))
add(cut["hook"]["from"] + 3.32, chime((1760, 2349, 2637), 0.05, 0.35), 0.8)
add(cut["hook"]["from"] + 3.4, thud(0.5), 0.35)
add(cut["hook"]["to"] - 0.35, whoosh(0.45, 300, 3500, up=True), 0.3)

# 3. The climb: the answer pops out of the roof.
add(cut["hook"]["to"] - 0.15 + 8.45, chime(), 0.5)

# 4-5. Investigate and test.
add(B[13], whoosh(1.2, 3000, 250), 0.35)
add(B[15] - 0.3, whoosh(0.4, 400, 5000, up=True), 0.35)
for name in ("attention", "votes", "steer", "puzzle"):
    for t in clip_events(name, {"click"}):
        add(t, boop(880, down=False), 0.5)
for k, t in enumerate(clip_events("unplug", {"click"})):
    add(t, boop([784, 659, 523][k]), 1.0)
add(B[20] + 0.15, boing(), 0.6)

# 6. The tour: a swish on every cut, keys for "featurama".
for b in B[22:29]:
    add(b - 0.12, whoosh(0.35, 5000, 800), 0.9)
for t in clip_events("tokens", {"key"}):
    add(t, key(), 0.9)

# 7. One last test: the backups, Mary falls, the crew lights up.
for t in clip_events("backups", {"click"}):
    add(t, boop(330), 0.8)
add(B[32] + 0.85, slide(), 0.55)
penta = [523, 587, 659, 784, 880, 1047, 1175, 1319, 1568, 1760, 2093]
for i, hz in enumerate(penta):
    add(B[34] + 0.1 + i * 0.25, boop(hz, down=False, d=0.09), 0.9)
add(beats["stop"] + 0.01, thud(), 2.6)

# 8. GPT-2 writes: a pluck as each word lands, a whoosh as the camera whips.
fill = next(b for b in B if b > beats["resume"] + 5)
per = (fill - beats["resume"]) / 6
for k, hz in enumerate([659, 784, 880, 988, 1175, 1319, 1568]):
    add(beats["resume"] + k * per + 0.78 * per, chime((hz, hz * 1.5), 0.03, 0.18), 0.6)
add(beats["resume"] + 6.8 * per, whoosh(beats["finalHit"] - beats["resume"] - 6.8 * per, 500, 4000, up=True), 0.5)
for i in range(len("shuw.github.io/featurama")):
    add(beats["finalHit"] + (30 + 1.5 * i + 1) / 30, key(), 0.6)

# Effects sit under the music: a gain of 1 peaks at under half the music's peak.
# A soft limit at the music's own peak keeps loud effects from ducking it.
peak = np.abs(music).max()
mix = music + fx * peak / 2.2
mix = np.tanh(mix / peak * 1.1) / np.tanh(1.1) * 0.93
sf.write(root / "public/trailer3-mix.wav", mix, SR)
print(f"sr {SR}, length {len(mix) / SR:.2f}s, fx peak {np.abs(fx).max():.2f}, music peak {np.abs(music).max():.2f}")
