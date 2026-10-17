"""The trailer's soundtrack: 60 seconds of electronic music at 128 BPM in
A minor, synthesized from scratch so it can be used anywhere.

Writes public/music.wav. The arrangement follows the video's sections
(src/timeline.ts counts the same beats):

  bars  1-2   intro: filtered pad and a rising sweep
  bars  3-6   build: kick, hats, rolling bass, arpeggio, filter opening
  bars  7-8   riser, snare roll, drum fill
  bars  9-20  drop: lead hook, chord stabs, full drums
  bars 21-22  break: drums out, pad and arpeggio, riser
  bars 23-30  second drop, lead an octave up
  bars 31-32  final chord hit and tail

Oscillators are band-limited (PolyBLEP), so they don't alias into harsh noise.
"""

import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt, sosfilt_zi

SR = 44100
BPM = 128
BEAT = 60 / BPM
BAR = 4 * BEAT
LENGTH = 60.0
N = int(SR * LENGTH)
NYQ = SR / 2
rng = np.random.default_rng(1939)
root = Path(__file__).resolve().parents[1]

# A minor: Am, F, C, G, one bar each.
CHORDS = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
BASS = [45, 41, 48, 43]
# The hook, in eighths over four bars: (eighth, MIDI note, length in eighths).
HOOK = [
    (0, 69, 1), (1, 72, 1), (2, 76, 2), (5, 74, 1), (6, 72, 2),
    (8, 69, 1), (9, 72, 1), (10, 77, 2), (13, 76, 1), (14, 72, 2),
    (16, 67, 1), (17, 72, 1), (18, 76, 2), (21, 79, 1), (22, 76, 2),
    (24, 74, 2), (26, 71, 1), (27, 67, 1), (28, 69, 3),
]

DRUMS = list(range(3, 21)) + list(range(23, 31))
DROP = list(range(9, 21)) + list(range(23, 31))


def hz(note):
    return 440 * 2 ** ((note - 69) / 12)


def t_of(bar, beat=0.0):
    return (bar - 1) * BAR + beat * BEAT


def env(n, attack, decay, sustain=0.0, release=None):
    t = np.arange(n) / SR
    a = np.minimum(1, t / max(attack, 1e-4))
    e = a * (sustain + (1 - sustain) * np.exp(-t / decay))
    if release:
        r = int(release * SR)
        if r < n:
            e[-r:] *= np.linspace(1, 0, r)
    return e


def blep_saw(ph, dt):
    """A saw with its jumps smoothed (PolyBLEP), so it doesn't alias."""
    y = 2 * ph - 1
    lo = ph < dt
    t = ph[lo] / dt
    y[lo] -= t + t - t * t - 1
    hi = ph > 1 - dt
    t = (ph[hi] - 1) / dt
    y[hi] -= t * t + t + t + 1
    return y


def osc(f, n, shape="saw", phase=0.0):
    """Band-limited saw or square."""
    dt = f / SR
    ph = (phase / (2 * np.pi) + dt * np.arange(n)) % 1
    if shape == "saw":
        return blep_saw(ph, dt)
    return 0.5 * (blep_saw(ph, dt) - blep_saw((ph + 0.5) % 1, dt))


def supersaw(f, n, voices=5, detune=0.012):
    out = np.zeros(n)
    for v in range(voices):
        d = 1 + detune * (v - (voices - 1) / 2) / ((voices - 1) / 2)
        out += osc(f * d, n, "saw", phase=rng.uniform(0, 2 * np.pi))
    return out / voices


def filt(x, cutoff, kind="low", order=4):
    sos = butter(order, np.clip(cutoff, 20, NYQ * 0.95) / NYQ, btype=kind, output="sos")
    return sosfilt(sos, x)


def sweep(x, cutoffs, kind="low", block=512):
    """Filter with a cutoff that changes over time (one value per block)."""
    out = np.empty_like(x)
    zi = None
    for i in range(0, len(x), block):
        c = cutoffs[min(len(cutoffs) - 1, i // block)]
        sos = butter(2, np.clip(c, 30, NYQ * 0.95) / NYQ, btype=kind, output="sos")
        if zi is None:
            zi = sosfilt_zi(sos) * x[0]
        out[i : i + block], zi = sosfilt(sos, x[i : i + block], zi=zi)
    return out


def add(track, start, sig, gain=1.0):
    i = int(round(start * SR))
    if i >= track.shape[-1] or i < 0:
        return
    m = min(sig.shape[-1], track.shape[-1] - i)
    track[..., i : i + m] += sig[..., :m] * gain


def reverb(x, seconds=2.2, wet=0.25, predelay=0.02):
    """Convolution with decaying noise, different on each side."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, x.shape[-1]))
    for side in range(2):
        ir = rng.standard_normal(n) * np.exp(-t / (seconds / 6.9))
        ir = filt(ir, 7000)
        ir[: int(predelay * SR)] = 0
        ir /= np.sqrt((ir**2).sum())
        src = x[side] if x.ndim == 2 else x
        out[side] = fftconvolve(src, ir)[: x.shape[-1]]
    dry = x if x.ndim == 2 else np.stack([x, x])
    return dry + wet * out


def delay(x, time, feedback=0.35, mix=0.3, pingpong=True):
    out = np.stack([x, x]) if x.ndim == 1 else x.copy()
    d = int(time * SR)
    echo = x if x.ndim == 1 else x.mean(0)
    g = mix
    for k in range(1, 6):
        side = k % 2 if pingpong else None
        shifted = np.zeros_like(echo)
        shifted[d * k :] = echo[: len(echo) - d * k] if d * k < len(echo) else 0
        if side is None:
            out += g * shifted
        else:
            out[side] += g * shifted
        g *= feedback
    return out


# --- Drums -----------------------------------------------------------------

def kick_hit():
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 48 + 140 * np.exp(-t / 0.028)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28)
    body = np.tanh(body * 1.6)
    click = filt(rng.standard_normal(n), 3000, "high") * np.exp(-t / 0.003) * 0.25
    return body + click


def clap_hit():
    n = int(0.4 * SR)
    noise = filt(filt(rng.standard_normal(n), 900, "high"), 6000)
    e = np.zeros(n)
    for k, off in enumerate((0, 0.009, 0.019, 0.03)):
        e += np.roll(env(n, 0.0008, 0.008 if k < 3 else 0.16), int(off * SR))
    snare = np.sin(2 * np.pi * 190 * np.arange(n) / SR) * env(n, 0.001, 0.05) * 0.4
    return noise * e + snare


def hat_hit(decay):
    n = int(decay * 7 * SR)
    noise = filt(rng.standard_normal(n), 8000, "high")
    return noise * env(n, 0.0005, decay)


def ride_hit():
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    metal = sum(np.sign(np.sin(2 * np.pi * f * t)) for f in (587, 845, 1253, 1673)) / 4
    return filt(metal + 0.5 * rng.standard_normal(n), 6000, "high") * env(n, 0.001, 0.18)


kick = np.zeros(N)
claps = np.zeros(N)
hats = np.zeros((2, N))
K, C, HC, HO, RIDE = kick_hit(), clap_hit(), hat_hit(0.022), hat_hit(0.11), ride_hit()
kick_times = []
for bar in DRUMS:
    fill = bar in (8, 20, 30)
    for b in range(4):
        if fill and b == 3:
            continue
        add(kick, t_of(bar, b), K)
        kick_times.append(t_of(bar, b))
        add(hats, t_of(bar, b + 0.5), np.stack([HO * 0.8, HO]) if bar in DROP else np.stack([HC, HC * 0.8]))
        if bar in DROP:
            add(hats, t_of(bar, b + 0.25), np.stack([HC * 0.5, HC * 0.3]))
            add(hats, t_of(bar, b + 0.75), np.stack([HC * 0.3, HC * 0.5]))
            add(hats, t_of(bar, b), np.stack([RIDE, RIDE]) * 0.18)
    if bar in DROP or bar >= 5:
        add(claps, t_of(bar, 1), C)
        add(claps, t_of(bar, 3), C)
    if fill:
        for s in (12, 13, 14, 15):
            add(claps, t_of(bar, s / 4), C * (0.5 + 0.12 * (s - 12)))

# Snare rolls into each drop, speeding up and rising.
for start in (7, 21, 29):
    steps = 48
    for i in range(steps):
        beat = 8 * (i / steps) ** 0.75
        if start == 29 and beat < 4:
            continue
        add(claps, t_of(start, beat), C * (0.2 + 0.8 * (i / steps) ** 1.5))

# Sidechain: everything melodic ducks under each kick.
duck = np.ones(N)
dn = int(0.3 * SR)
shape = 1 - 0.82 * np.exp(-np.arange(dn) / SR / 0.07)
for t in kick_times:
    i = int(t * SR)
    m = min(dn, N - i)
    duck[i : i + m] = np.minimum(duck[i : i + m], shape[:m])

# --- Bass ------------------------------------------------------------------

bass = np.zeros(N)
sub = np.zeros(N)
for bar in DRUMS:
    note = BASS[(bar - 1) % 4]
    for s in range(16):
        if s % 4 == 0:
            continue
        n = int(BEAT / 4 * SR * 0.95)
        up = 12 if bar in DROP and s in (6, 14) else 0
        f = hz(note + up)
        tone = osc(f, n, "saw")
        # A filter that snaps shut on each note: the pluck.
        cut = 300 + 2600 * np.exp(-np.arange(n) / SR / 0.035)
        tone = sweep(tone, cut[::256], block=256)
        add(bass, t_of(bar, s / 4), tone * env(n, 0.002, 0.12, 0.3, 0.01))
    n = int(BAR * SR)
    add(sub, t_of(bar), np.sin(2 * np.pi * hz(note - 12) * np.arange(n) / SR) * env(n, 0.01, 10, 1, 0.05))
bass = filt(bass, 40, "high", 2)

# --- Pads, stabs, arpeggio, lead ------------------------------------------

pad = np.zeros((2, N))
for bar in range(1, 31):
    chord = CHORDS[(bar - 1) % 4]
    n = int(BAR * SR + 0.4 * SR)
    voicing = chord + [chord[0] - 12, chord[1] + 12]
    for side in range(2):
        sig = sum(supersaw(hz(m), n, 5, 0.014 + 0.004 * side) for m in voicing) / len(voicing)
        add(pad[side], t_of(bar), sig * env(n, 0.3 if bar in (1, 21) else 0.08, 4, 0.7, 0.3))
# The pad opens up through the intro and build, and again in the break.
cut = np.full(N // 512 + 1, 2400.0)
for a, b, lo, hi in ((1, 9, 300, 2400), (21, 23, 500, 2400)):
    i0, i1 = int(t_of(a) * SR) // 512, int(t_of(b) * SR) // 512
    cut[i0:i1] = np.geomspace(lo, hi, i1 - i0)
for side in range(2):
    pad[side] = filt(sweep(pad[side], cut), 180, "high", 2)

stabs = np.zeros((2, N))
for bar in DROP:
    chord = CHORDS[(bar - 1) % 4]
    for b in range(4):
        n = int(BEAT * 0.45 * SR)
        sig = sum(supersaw(hz(m + 12), n, 5, 0.018) for m in chord) / 3
        sig = filt(sig * env(n, 0.002, 0.09), 5000)
        add(stabs, t_of(bar, b + 0.5), np.stack([sig, sig * 0.9]))
stabs = reverb(stabs, 1.4, 0.2)

arp = np.zeros(N)
for bar in list(range(3, 21)) + list(range(21, 31)):
    chord = CHORDS[(bar - 1) % 4]
    notes = [m + 12 * o for o in (1, 2) for m in chord]
    for s in range(16):
        n = int(0.14 * SR)
        f = hz(notes[(s * 2 + s // 4) % len(notes)])
        add(arp, t_of(bar, s / 4), osc(f, n, "square") * env(n, 0.001, 0.045) * (1.0 if bar in DROP else 0.8))
arp = filt(filt(arp, 4500), 300, "high", 2)
arp = delay(arp, BEAT * 0.75, 0.4, 0.35)

lead = np.zeros(N)
for start in list(range(9, 21, 4)) + list(range(23, 31, 4)):
    up = 12 if start >= 23 else 0
    for eighth, note, length in HOOK:
        n = int(length * BEAT / 2 * SR * 0.92)
        f = hz(note + up)
        tone = supersaw(f, n, 7, 0.016) + 0.5 * supersaw(f * 2, n, 3, 0.01)
        add(lead, t_of(start, eighth / 2), tone * env(n, 0.004, 0.25, 0.65, 0.04))
lead = filt(filt(lead, 6500), 250, "high", 2)
lead = reverb(delay(lead, BEAT * 0.75, 0.3, 0.22), 2.4, 0.3)

# --- Effects -----------------------------------------------------------------

fx = np.zeros((2, N))
for start, bars in ((1, 2), (7, 2), (21, 2), (29, 2)):
    n = int(bars * BAR * SR)
    p = np.linspace(0, 1, n)
    noise = rng.standard_normal(n)
    noise = sweep(noise, np.geomspace(400, 9000, n // 512 + 1), "high")
    up = np.sin(2 * np.pi * np.cumsum(220 * 2 ** (3 * p**2)) / SR) * 0.25
    sig = (noise * 0.5 + up) * p**2.2
    add(fx, t_of(start), np.stack([sig, np.roll(sig, 300)]))


def impact(size=1.0):
    n = int(3 * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(36 + 70 * np.exp(-t / 0.07)) / SR) * np.exp(-t / 0.8)
    crash = filt(rng.standard_normal(n), 3500, "high") * np.exp(-t / 1.1) * 0.35
    return (boom + crash) * size


for bar, size in ((3, 0.6), (9, 0.9), (23, 0.9), (31, 1.2)):
    s = impact(size)
    add(fx, t_of(bar), np.stack([s, s]))
fx = reverb(fx, 2.5, 0.2)

# The final chord: everything at once, left to ring.
final = np.zeros((2, N))
n = int(4 * SR)
chord = [45, 57, 60, 64, 69, 72, 76]
hit = sum(supersaw(hz(m), n, 7, 0.02) for m in chord) / len(chord) * env(n, 0.002, 1.2, 0.0)
hit = filt(hit, 5000)
add(final, t_of(31), np.stack([hit, hit]))
final = reverb(final, 3.5, 0.45)
add(kick, t_of(31), K * 1.2)

# --- Mix -----------------------------------------------------------------------

# The build sits lower than the drops, so each drop lands.
lift = np.ones(N)
i0, i1 = int(t_of(3) * SR), int(t_of(9) * SR)
lift[i0:i1] = np.linspace(0.5, 0.85, i1 - i0)
mono = lambda x: np.stack([x, x])

music = (
    mono(bass * 0.55 + sub * 0.35) * duck
    + pad * 0.32 * duck
    + stabs * 0.45 * duck
    + arp * 0.35 * duck
    + lead * 0.55 * duck
)
drums = mono(kick * 0.95) + reverb(mono(claps), 1.2, 0.18) * 0.42 + hats * 0.16
mix = (music + drums) * lift + fx * 0.5 + final * 0.8

# After the final hit, only the hit rings on.
tail = int((t_of(31) + 0.05) * SR)
keep = (fx * 0.5 + final * 0.8)[:, tail:] + mono(kick * 0.95)[:, tail:]
mix[:, tail:] = keep * np.exp(-(np.arange(N - tail) / SR) / 1.6)

# Master: glue compression by a slow gain follower, then soft clipping.
smooth = np.convolve(np.abs(mix).mean(0), np.ones(2205) / 2205, mode="same")
gain = 1 / (1 + np.maximum(0, smooth / (np.percentile(smooth, 90) + 1e-9) - 1) * 0.5)
mix *= gain
mix /= np.abs(mix).max() / 1.3
mix = np.tanh(mix) / np.tanh(1.3)
mix /= np.abs(mix).max() / 0.95
fade = int(0.02 * SR)
mix[:, :fade] *= np.linspace(0, 1, fade)
mix[:, -int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))

out = root / "public" / "music.wav"
pcm = (mix.T * 32767).astype("<i2")
with wave.open(str(out), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"wrote {out} ({LENGTH:.0f} s, {BPM} BPM)")
