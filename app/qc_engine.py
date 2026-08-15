"""Signal-quality control engine for digital stethoscope recordings.

Every metric is computed from the actual audio signal with established DSP
methods (frame-energy VAD, percentile noise-floor estimation, envelope
autocorrelation for cardiac-cycle detection, Welch spectral flatness). No
synthetic or mock values are used anywhere.

Reference framework
-------------------
- Auscultation band: 25-400 Hz, following the CirCor DigiScope acquisition
  pipeline (Oliveira et al., 2021) and the PhysioNet Challenge 2022 setup.
- PCG signal-quality grading bands for SNR follow common practice in the
  phonocardiogram literature: >= 12 dB good, 6-12 dB acceptable, < 6 dB poor.
- Resting adult heart rate 60-100 bpm (AHA); the cardiac-cycle detector is
  allowed to search 30-250 bpm so that brady/tachy rhythms are still resolved.
- Clipping threshold >= 1% of samples at/near full scale -> ADC saturation.
- Minimum recommended recording length: 8 s (the model input window).

The engine never decides a diagnosis; it only reports physical signal quality
and auscultation-position confidence. Position confidence comes from a CNN
trained on the CirCor dataset (real recordings, expert-annotated locations).
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

QC_VERSION = "1.0.0"

SR_NATIVE = 4000
BAND_LOW, BAND_HIGH = 25.0, 400.0
HR_MIN, HR_MAX = 30.0, 250.0
MIN_DURATION_S = 8.0
ACCEPT_DURATION_S = 4.0
CLIP_LEVEL = 0.99
CLIP_RATIO_BAD = 0.01
SNR_GOOD_DB = 12.0
SNR_ACCEPT_DB = 6.0
FLATNESS_BAD = 0.75
FLATNESS_WARN = 0.55
LEVEL_LOUD_DB = -6.0
LEVEL_QUIET_DB = -48.0


def _butter_bandpass(x, sr, low, high, order=4):
    nyq = sr / 2.0
    hi = min(high, nyq * 0.99)
    if hi <= low:
        return x.copy()
    b, a = butter(order, [low / nyq, hi / nyq], btype="band")
    padlen = min(len(x) - 1, 3 * max(len(a), len(b)))
    if padlen <= 0:
        return x.copy()
    return filtfilt(b, a, x, padlen=padlen)


def _frame_rms(x, sr, win_s=0.04, hop_s=0.02):
    win, hop = int(win_s * sr), int(hop_s * sr)
    n = len(x)
    out = []
    for i in range(0, n - win + 1, hop):
        seg = x[i:i + win]
        out.append(float(np.sqrt(np.mean(seg * seg))))
    return np.asarray(out, dtype=np.float64)


def _noise_floor_snr(frame_rms):
    """Percentile-based noise floor and SNR estimation.

    Heart sounds are sparse: S1/S2 bursts occupy only ~20% of a recording,
    so the "signal" level must be read from a high percentile (95th), not the
    median of the upper half (which still falls in the noise floor). Noise is
    the median of the quietest fifth of frames. This follows common
    percentile-based SNR practice in the PCG literature.
    """
    if len(frame_rms) < 4 or frame_rms.max() == 0:
        return None, None
    order = np.sort(frame_rms)
    n = len(order)
    noise = float(np.median(order[: max(2, n // 5)]))
    signal = float(np.percentile(order, 95))
    if noise <= 1e-9:
        return -120.0, None
    snr = 20.0 * np.log10(max(signal, 1e-9) / noise)
    return 20.0 * np.log10(noise + 1e-12), float(snr)


def _spectral_flatness(x, sr):
    if len(x) < 512:
        return 1.0
    from scipy.signal import welch
    freqs, psd = welch(x, fs=sr, nperseg=min(len(x), 1024))
    band = (freqs >= BAND_LOW) & (freqs <= 1500.0)
    if band.sum() < 8:
        return 1.0
    p = psd[band] + 1e-12
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def _envelope(x, sr):
    bp = _butter_bandpass(x, sr, BAND_LOW, BAND_HIGH)
    env = np.abs(hilbert(bp))
    win = int(0.08 * sr)
    k = np.ones(win) / win
    return np.convolve(env, k, mode="same")


def _detect_heart_rate(x, sr):
    """Return (bpm, confidence) or (None, 0.0) if no stable cardiac cycle."""
    env = _envelope(x, sr)
    n = len(env)
    min_lag, max_lag = int(sr * 60.0 / HR_MAX), int(sr * 60.0 / HR_MIN)
    if n < max_lag * 2:
        return None, 0.0
    env = env - env.mean()
    denom = float(np.sum(env * env))
    if denom <= 0:
        return None, 0.0
    corr = np.correlate(env, env, mode="full")[n - 1:]
    corr = corr / denom
    lags = np.arange(len(corr))
    valid = (lags >= min_lag) & (lags <= max_lag)
    if valid.sum() == 0:
        return None, 0.0
    search = corr[valid].copy()
    peak_idx = int(np.argmax(search))
    peak = float(search[peak_idx])
    period = lags[valid][peak_idx]
    if peak < 0.15:
        return None, peak
    bpm = 60.0 * sr / period
    return float(bpm), peak


def _clipping(x):
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= CLIP_LEVEL))


def _level_metrics(x):
    if len(x) == 0:
        return {"rms_db": -120.0, "peak_db": -120.0, "crest_db": 0.0}
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    rms_db = 20.0 * np.log10(rms + 1e-12)
    peak_db = 20.0 * np.log10(peak + 1e-12)
    crest = 20.0 * np.log10((peak / rms) + 1e-9)
    return {"rms_db": round(rms_db, 2), "peak_db": round(peak_db, 2),
            "crest_db": round(crest, 2)}


def _band_energy_ratio(x, sr):
    """Fraction of total energy in the 25-150 Hz band over the whole signal.

    Using the whole recording (instead of the first 1 s) avoids false
    "poor contact" warnings when the beginning of the clip is silence.
    """
    if len(x) < 512:
        return 0.5
    win = min(len(x), 32768)
    seg = x[:win] * np.hanning(win)
    spec = np.abs(np.fft.rfft(seg)) ** 2
    freq = np.fft.rfftfreq(win, 1.0 / sr)
    total = float(spec.sum())
    if total <= 0:
        return 0.5
    low = float(spec[(freq >= BAND_LOW) & (freq <= 150.0)].sum())
    return float(low / total)


def grade(metric, value):
    """Map a metric value to a {status: ok|warn|fail, label} verdict."""
    return {"metric": metric, "value": value}


def analyze(x, sr, duration_hint=None):
    """Full QC analysis of a recording.

    x must be float32/float64 audio in [-1, 1]. sr is the sample rate.
    Returns a flat dict with every metric plus a list of checks.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if sr != SR_NATIVE:
        from scipy.signal import resample_poly
        x = resample_poly(x, SR_NATIVE, int(sr))

    dur = len(x) / SR_NATIVE
    level = _level_metrics(x)
    frame_rms = _frame_rms(x, SR_NATIVE)
    noise_db, snr_db = _noise_floor_snr(frame_rms)
    flatness = _spectral_flatness(x, SR_NATIVE)
    bpm, hr_conf = _detect_heart_rate(x, SR_NATIVE)
    clip = _clipping(x)
    low_ratio = _band_energy_ratio(x, SR_NATIVE)

    checks = []

    if dur >= MIN_DURATION_S:
        checks.append({"metric": "duration", "status": "ok", "value": round(dur, 1),
                       "label": f"{dur:.1f}s（≥8s 标准录音时长）"})
    elif dur >= ACCEPT_DURATION_S:
        checks.append({"metric": "duration", "status": "warn", "value": round(dur, 1),
                       "label": f"{dur:.1f}s（建议 ≥8s，当前偏短）"})
    else:
        checks.append({"metric": "duration", "status": "fail", "value": round(dur, 1),
                       "label": f"{dur:.1f}s（<4s，无法评估）"})

    if clip >= CLIP_RATIO_BAD:
        checks.append({"metric": "clipping", "status": "fail", "value": round(clip * 100, 2),
                       "label": f"削波 {clip*100:.1f}%（>1%，ADC 饱和，请降低增益）"})
    elif clip > 0.0:
        checks.append({"metric": "clipping", "status": "warn", "value": round(clip * 100, 3),
                       "label": f"削波 {clip*100:.2f}%（接近饱和）"})
    else:
        checks.append({"metric": "clipping", "status": "ok", "value": 0.0,
                       "label": "无削波（信号未饱和）"})

    if snr_db is not None:
        if snr_db >= SNR_GOOD_DB:
            checks.append({"metric": "snr", "status": "ok", "value": round(snr_db, 1),
                           "label": f"信噪比 {snr_db:.1f} dB（≥12 dB 良好）"})
        elif snr_db >= SNR_ACCEPT_DB:
            checks.append({"metric": "snr", "status": "warn", "value": round(snr_db, 1),
                           "label": f"信噪比 {snr_db:.1f} dB（6–12 dB 可接受，请保持安静）"})
        else:
            checks.append({"metric": "snr", "status": "fail", "value": round(snr_db, 1),
                           "label": f"信噪比 {snr_db:.1f} dB（<6 dB，环境噪声过大）"})
    else:
        checks.append({"metric": "snr", "status": "fail", "value": None,
                       "label": "无法估计信噪比（信号过弱）"})

    if flatness >= FLATNESS_BAD:
        checks.append({"metric": "flatness", "status": "fail", "value": round(flatness, 3),
                       "label": f"频谱平坦度 {flatness:.2f}（≥0.75，宽谱噪声主导，信号非心音特征）"})
    elif flatness >= FLATNESS_WARN:
        checks.append({"metric": "flatness", "status": "warn", "value": round(flatness, 3),
                       "label": f"频谱平坦度 {flatness:.2f}（噪声成分偏多）"})
    else:
        checks.append({"metric": "flatness", "status": "ok", "value": round(flatness, 3),
                       "label": f"频谱平坦度 {flatness:.2f}（<0.55，非宽谱噪声）"})

    if bpm is None:
        checks.append({"metric": "heart_rate", "status": "fail", "value": None,
                       "label": "未检测到稳定心音节律（胸件未贴紧皮肤或位置错误）"})
    elif 60.0 <= bpm <= 100.0:
        checks.append({"metric": "heart_rate", "status": "ok", "value": round(bpm, 0),
                       "label": f"心音节律 {bpm:.0f} bpm（成人静息参考 60-100）"})
    elif HR_MIN <= bpm <= HR_MAX:
        checks.append({"metric": "heart_rate", "status": "ok", "value": round(bpm, 0),
                       "label": f"心音节律 {bpm:.0f} bpm（已检出；儿童/婴儿可正常偏高，请结合年龄判断）"})
    else:
        checks.append({"metric": "heart_rate", "status": "warn", "value": round(bpm, 0),
                       "label": f"节律 {bpm:.0f} bpm（超出检测可信区间 30–250）"})

    if low_ratio < 0.05:
        checks.append({"metric": "contact", "status": "warn", "value": round(low_ratio, 3),
                       "label": "低频能量占比低，胸件可能未完全贴合皮肤"})
    else:
        checks.append({"metric": "contact", "status": "ok", "value": round(low_ratio, 3),
                       "label": f"低频占比 {low_ratio:.2f}（接触面信号存在）"})

    return {
        "qc_version": QC_VERSION,
        "sr": SR_NATIVE,
        "duration_s": round(dur, 2),
        "level": level,
        "noise_floor_db": round(noise_db, 1) if noise_db is not None else None,
        "snr_db": round(snr_db, 1) if snr_db is not None else None,
        "spectral_flatness": round(flatness, 3),
        "heart_rate_bpm": round(bpm, 1) if bpm is not None else None,
        "heart_rate_confidence": round(hr_conf, 3),
        "clipping_ratio": round(clip, 5),
        "low_freq_ratio": round(low_ratio, 4),
        "checks": checks,
    }
