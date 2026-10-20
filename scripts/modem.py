"""A JANUS-like frequency-hopped binary FSK frame with a rate-1/2 K=9 convolutional code.

Structure follows STANAG 4748 JANUS in outline: 13 hop slots, each a pair of tones 160 Hz apart, 160 chips per
second (6.25 ms chip), 4160 Hz total band, 64-bit payload plus 8 tail bits encoded by the rate-1/2 K=9 code
with generators (0o753, 0o561), 32-chip known preamble. Carrier placement is a parameter so the same frame fits
each library channel's band (Blue 10.5-15.5 kHz, Red 20-30 kHz, Black about 11.75-24.25 kHz, Green 3.75-8.25 kHz).
Detection is non-coherent (energy at the two tones over the chip window), soft-decision Viterbi.

A `Rate` object raises the chip rate inside the same 4160 Hz band (320, 640, 1920 chips/s with 6, 3 and 1 hop
slots), which shortens the frame and the hop reuse interval. That is the rate ladder used to find the goodput
ceiling a measured channel supports.

The receiver knows the hop pattern and finds chip timing from the preamble by searching start offsets up to
`max_delay_s`. Nothing here is tuned to a channel.
"""
from __future__ import annotations

import numpy as np

CHIP_RATE = 160.0
TONE_SPACING = 160.0
N_SLOTS = 13
BAND_HZ = 26 * TONE_SPACING  # 4160 Hz
PAYLOAD_BITS = 64
TAIL_BITS = 8
PREAMBLE_CHIPS = 32
G1, G2 = 0o753, 0o561
K = 9

# Fixed hop patterns: permutations with consecutive slots at least two apart where the slot count allows.
HOP_PATTERNS = {13: np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5, 12]), 6: np.array([0, 3, 1, 4, 2, 5]),
                3: np.array([0, 2, 1]), 1: np.array([0])}
_rng_pre = np.random.default_rng(4748)
PREAMBLE_BITS = _rng_pre.integers(0, 2, PREAMBLE_CHIPS)


class Rate:
    """Frame rate configuration inside the fixed 4160 Hz band: chip rate = tone spacing, slots = 4160 // (2 rate)."""

    def __init__(self, chip_rate: float = CHIP_RATE):
        self.chip_rate = float(chip_rate)
        self.n_slots = int(BAND_HZ // (2 * self.chip_rate))
        if self.n_slots not in HOP_PATTERNS:
            raise ValueError(f"no hop pattern for {self.n_slots} slots")
        self.hop = HOP_PATTERNS[self.n_slots]
        self.band_used = 2 * self.n_slots * self.chip_rate

    def tone_freqs(self, fc: float) -> np.ndarray:
        n_tones = 2 * self.n_slots
        return fc - self.band_used / 2 + self.chip_rate / 2 + self.chip_rate * np.arange(n_tones)

    @property
    def bits_per_s(self) -> float:
        """Payload bits over the frame duration, preamble included (the same definition as coherent.Scheme.bits_per_s,
        so the two frame families are compared on one basis; 58.2 bit/s at 160 chips/s). Until 2026-10-20 this was
        chip_rate / 2, which excluded the 32-chip preamble (CORRECTIONS C4). The replay guard of 0.35 s is not counted
        in either family: it is a protocol choice a system would set from the delay spread."""
        return PAYLOAD_BITS / self.frame_duration_s()

    def frame_duration_s(self) -> float:
        return (PREAMBLE_CHIPS + 2 * (PAYLOAD_BITS + TAIL_BITS)) / self.chip_rate


DEFAULT_RATE = Rate(CHIP_RATE)


def conv_encode(bits: np.ndarray) -> np.ndarray:
    """Rate-1/2, K=9 convolutional code, zero-terminated by the caller (bits already include tail)."""
    state = 0
    out = np.empty(2 * bits.size, dtype=np.int8)
    for i, b in enumerate(bits):
        state = ((state << 1) | int(b)) & ((1 << K) - 1)
        out[2 * i] = bin(state & G1).count("1") & 1
        out[2 * i + 1] = bin(state & G2).count("1") & 1
    return out


def _trellis():
    n_states = 1 << (K - 1)
    nxt = np.zeros((n_states, 2), dtype=np.int32)
    outs = np.zeros((n_states, 2, 2), dtype=np.float64)
    for s in range(n_states):
        for b in (0, 1):
            full = ((s << 1) | b) & ((1 << K) - 1)
            nxt[s, b] = full & (n_states - 1)
            outs[s, b, 0] = 1 - 2 * (bin(full & G1).count("1") & 1)  # +1 for coded bit 0, -1 for bit 1
            outs[s, b, 1] = 1 - 2 * (bin(full & G2).count("1") & 1)
    return nxt, outs


NXT, OUTS = _trellis()


def viterbi_decode(soft: np.ndarray, n_bits: int) -> np.ndarray:
    """Soft-decision Viterbi. `soft` has 2*n_bits entries, positive means coded bit 0 more likely.

    Terminated trellis (tail bits force state 0), start in state 0. Vectorised add-compare-select: the two
    predecessors of state d are p0 = d >> 1 and p1 = p0 | 128, both with input bit b = d & 1.
    """
    n_states = NXT.shape[0]
    dest = np.arange(n_states)
    b = dest & 1
    p0 = dest >> 1
    p1 = p0 | (n_states >> 1)
    o0 = OUTS[p0, b, :]
    o1 = OUTS[p1, b, :]
    metric = np.full(n_states, -np.inf)
    metric[0] = 0.0
    back = np.zeros((n_bits, n_states), dtype=np.int32)
    for i in range(n_bits):
        sv = soft[2 * i:2 * i + 2]
        c0 = metric[p0] + o0 @ sv
        c1 = metric[p1] + o1 @ sv
        take1 = c1 > c0
        metric = np.where(take1, c1, c0)
        back[i] = np.where(take1, p1, p0)
    s = 0
    bits = np.zeros(n_bits, dtype=np.int8)
    for i in range(n_bits - 1, -1, -1):
        bits[i] = s & 1
        s = back[i, s]
    return bits


def tone_freqs(fc: float, rate: Rate = DEFAULT_RATE) -> np.ndarray:
    """Tones `rate.chip_rate` Hz apart, centred on fc. Slot s uses tones (2s, 2s+1)."""
    return rate.tone_freqs(fc)


def frame_chips(payload: np.ndarray) -> np.ndarray:
    """Preamble chips followed by coded data chips."""
    bits = np.concatenate([payload.astype(np.int8), np.zeros(TAIL_BITS, dtype=np.int8)])
    coded = conv_encode(bits)
    return np.concatenate([PREAMBLE_BITS.astype(np.int8), coded])


def modulate(chips: np.ndarray, fc: float, fs: float, rate: Rate = DEFAULT_RATE) -> np.ndarray:
    """Unit-power real passband FH-BFSK signal."""
    n = int(round(fs / rate.chip_rate))
    freqs = tone_freqs(fc, rate)
    t = np.arange(n) / fs
    sig = np.empty(chips.size * n)
    phase = 0.0
    for i, c in enumerate(chips):
        slot = rate.hop[i % rate.n_slots]
        f = freqs[2 * slot + int(c)]
        sig[i * n:(i + 1) * n] = np.cos(2 * np.pi * f * t + phase)
        phase = (phase + 2 * np.pi * f * n / fs) % (2 * np.pi)
    return sig * np.sqrt(2.0)  # unit mean-square


def frame_duration_s(rate: Rate = DEFAULT_RATE) -> float:
    return rate.frame_duration_s()


def _chip_energies(x: np.ndarray, fc: float, fs: float, start: int, n_chips: int, chip0: int = 0,
                   rate: Rate = DEFAULT_RATE) -> np.ndarray:
    """Energy at the two tones of each chip's slot. `chip0` is the frame index of the first chip. Returns (n_chips, 2)."""
    n = int(round(fs / rate.chip_rate))
    freqs = tone_freqs(fc, rate)
    seg = x[start:start + n_chips * n]
    if seg.size < n_chips * n:
        seg = np.concatenate([seg, np.zeros(n_chips * n - seg.size)])
    seg = seg.reshape(n_chips, n)
    t = np.arange(n) / fs
    e = np.empty((n_chips, 2))
    for i in range(n_chips):
        slot = rate.hop[(chip0 + i) % rate.n_slots]
        for b in (0, 1):
            f = freqs[2 * slot + b]
            z = np.dot(seg[i], np.exp(-2j * np.pi * f * t))
            e[i, b] = np.abs(z) ** 2
    return e


def _preamble_metric_vec(x: np.ndarray, fc: float, fs: float, offsets: np.ndarray, rate: Rate = DEFAULT_RATE) -> np.ndarray:
    """Preamble correlation metric for many candidate start offsets at once."""
    n = int(round(fs / rate.chip_rate))
    freqs = tone_freqs(fc, rate)
    t = np.arange(n) / fs
    xp = np.concatenate([x, np.zeros(int(offsets.max()) + PREAMBLE_CHIPS * n + n)])
    metric = np.zeros(offsets.size)
    base = np.arange(n)[None, :]
    for i in range(PREAMBLE_CHIPS):
        slot = rate.hop[i % rate.n_slots]
        seg = xp[offsets[:, None] + i * n + base]
        E = np.exp(-2j * np.pi * np.outer(t, freqs[2 * slot:2 * slot + 2]))  # (n, 2)
        e = np.abs(seg @ E) ** 2
        tx = int(PREAMBLE_BITS[i])
        metric += (e[:, tx] - e[:, 1 - tx]) / (e[:, 0] + e[:, 1] + 1e-12)
    return metric


def demodulate(x: np.ndarray, fc: float, fs: float, max_delay_s: float = 0.30, step_s: float = 0.0004,
               rate: Rate = DEFAULT_RATE):
    """Return (decoded payload bits, start offset in samples, preamble metric)."""
    n = int(round(fs / rate.chip_rate))
    step = max(1, min(n // 4, int(step_s * fs)))
    offsets = np.arange(0, int(max_delay_s * fs), step)
    m = _preamble_metric_vec(x, fc, fs, offsets, rate)
    best = offsets[int(np.argmax(m))]
    n_data = 2 * (PAYLOAD_BITS + TAIL_BITS)
    e = _chip_energies(x, fc, fs, best + PREAMBLE_CHIPS * n, n_data, chip0=PREAMBLE_CHIPS, rate=rate)
    soft = (e[:, 0] - e[:, 1]) / (e[:, 0] + e[:, 1] + 1e-12)
    bits = viterbi_decode(soft, PAYLOAD_BITS + TAIL_BITS)
    return bits[:PAYLOAD_BITS], int(best), float(np.max(m))


def awgn_selftest(snr_db_list=(-14, -12, -10, -8, -6, -4), n_frames=60, fc=13000.0, fs=96000.0, seed=1,
                  rate: Rate = DEFAULT_RATE):
    """Packet success rate in AWGN, SNR defined as signal power over noise power in the 4160 Hz frame band."""
    rng = np.random.default_rng(seed)
    n = int(round(fs / rate.chip_rate))
    res = {}
    for snr_db in snr_db_list:
        ok = 0
        for _ in range(n_frames):
            payload = rng.integers(0, 2, PAYLOAD_BITS)
            x = modulate(frame_chips(payload), fc, fs, rate)
            delay = rng.integers(0, int(0.1 * fs))
            x = np.concatenate([np.zeros(delay), x, np.zeros(n * 4)])
            noise_psd = 1.0 / (10 ** (snr_db / 10) * BAND_HZ)  # white noise with P_sig / snr inside the band
            w = rng.standard_normal(x.size) * np.sqrt(noise_psd * fs / 2)
            bits, _, _ = demodulate(x + w, fc, fs, rate=rate)
            ok += int(np.array_equal(bits, payload))
        res[snr_db] = ok / n_frames
        print(f"AWGN rate {rate.chip_rate:.0f} chip/s SNR {snr_db:+5.1f} dB in-band: PSR {res[snr_db]:.3f}")
    return res


if __name__ == "__main__":
    import time
    for cr in (160, 320, 640, 1920):
        r = Rate(cr)
        t0 = time.time()
        print("rate", cr, "slots", r.n_slots, "frame", round(r.frame_duration_s(), 3), "s; bits/s", r.bits_per_s)
        awgn_selftest(snr_db_list=(-12, -10, -8, -6, -4, -2, 0), n_frames=30, rate=r)
        print("  ", round((time.time() - t0) / 210, 3), "s per frame")
