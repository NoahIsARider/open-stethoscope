// mel.js — librosa-1.0.0-equivalent preprocessing for the Open Stethoscope
// browser demo. Mirrors demo/browser/tools/mel_np.py exactly:
//   zero-pad (n_fft/2 each side) → frame (hop) → hann window → rfft →
//   power → mel filterbank → power_to_db(ref=max, top_db=80) →
//   trim/pad to n_frames.
// Pure JS, no DOM — testable in node.

// ─── param loading (shared by browser and node) ────────────────────────────
// mel_params.json stores the filterbank as nested rows (as numpy sees it);
// flatten to row-major Float32Array for the JS math.
export function loadParams(json) {
  return {
    ...json,
    window: new Float32Array(json.window),
    filterbank: new Float32Array(json.filterbank.flat()),
  }
}

// ─── radix-2 iterative FFT (in place) ──────────────────────────────────────
function fft(re, im) {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t
      t = im[i]; im[i] = im[j]; im[j] = t
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len
    const wRe = Math.cos(ang), wIm = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0
      const half = len >> 1
      for (let k = 0; k < half; k++) {
        const uRe = re[i + k], uIm = im[i + k]
        const vRe = re[i + k + half] * curRe - im[i + k + half] * curIm
        const vIm = re[i + k + half] * curIm + im[i + k + half] * curRe
        re[i + k] = uRe + vRe
        im[i + k] = uIm + vIm
        re[i + k + half] = uRe - vRe
        im[i + k + half] = uIm - vIm
        const nRe = curRe * wRe - curIm * wIm
        curIm = curRe * wIm + curIm * wRe
        curRe = nRe
      }
    }
  }
}

// ─── mel spectrogram ───────────────────────────────────────────────────────
// x: Float32Array mono audio at p.sr Hz → Float32Array(n_mels * nFrames)
export function melSpectrogram(x, p) {
  const nFft = p.n_fft, hop = p.hop, nMels = p.n_mels
  const win = p.window            // Float32Array(nFft)
  const fb = p.filterbank         // Float32Array(nMels * (nFft/2+1)), row-major
  const nBins = nFft / 2 + 1

  // zero-pad nFft/2 on each side (librosa center=True, pad_mode='constant')
  const N = x.length + nFft
  const nFrames = 1 + Math.floor((N - nFft) / hop)

  const re = new Float32Array(nFft)
  const im = new Float32Array(nFft)
  const mel = new Float32Array(nMels * nFrames)

  let maxPower = 1e-30
  for (let t = 0; t < nFrames; t++) {
    const start = t * hop - nFft / 2
    for (let i = 0; i < nFft; i++) {
      const s = start + i
      const v = s >= 0 && s < x.length ? x[s] : 0
      re[i] = v * win[i]
      im[i] = 0
    }
    fft(re, im)
    for (let b = 0; b < nBins; b++) {
      const pw = re[b] * re[b] + im[b] * im[b]
      for (let m = 0; m < nMels; m++) {
        const w = fb[m * nBins + b]
        if (w !== 0) mel[m * nFrames + t] += pw * w
      }
    }
  }
  // track max for power_to_db
  for (let i = 0; i < mel.length; i++) if (mel[i] > maxPower) maxPower = mel[i]
  const ref = Math.max(maxPower, 1e-10)
  const floorDb = -80 // librosa power_to_db top_db=80 → clip at max(0dB) − 80
  for (let i = 0; i < mel.length; i++) {
    let db = 10 * Math.log10(Math.max(mel[i], 1e-10) / ref)
    if (db < floorDb) db = floorDb
    mel[i] = db
  }
  return mel
}

// trim to the first n_frames of *each* row, or zero-pad rows that are short
// (mirrors numpy `m[:, :n_frames]` + right-pad)
export function toFrames(mel, nFrames, nMels) {
  const have = mel.length / nMels
  if (have === nFrames) return mel
  const out = new Float32Array(nMels * nFrames)
  for (let m = 0; m < nMels; m++) {
    const src = m * have, dst = m * nFrames
    const n = Math.min(nFrames, have)
    out.set(mel.subarray(src, src + n), dst)
  }
  return out
}

// build model input: x (1,4,1,n_mels,n_frames) + bool mask for one location
export function buildInput(melFrames, locIdx, p) {
  const nFrames = p.n_frames, nMels = p.n_mels
  const xb = new Float32Array(4 * 1 * nMels * nFrames)
  const base = locIdx * nMels * nFrames
  xb.set(melFrames, base)
  const mask = new Uint8Array(4)
  mask[locIdx] = 1
  return { xb, mask }
}

// linear resample to p.sr (matches the verify script's np.interp)
export function resampleTo(x, srcSr, dstSr) {
  if (srcSr === dstSr) return x
  const outLen = Math.floor((x.length * dstSr) / srcSr)
  const out = new Float32Array(outLen)
  const step = srcSr / dstSr
  for (let i = 0; i < outLen; i++) {
    const pos = i * step
    const i0 = Math.floor(pos)
    const i1 = Math.min(i0 + 1, x.length - 1)
    const frac = pos - i0
    out[i] = x[i0] * (1 - frac) + x[i1] * frac
  }
  return out
}

// decode an ArrayBuffer to mono float32 at a target sample rate
// (works in both browser and node via a provided decode function)
export async function decodeToMono(arrayBuffer, decodeAudio) {
  const ctx = new (globalThis.OfflineAudioContext || globalThis.AudioContext)(1, 1, 48000)
  const audioBuf = await ctx.decodeAudioData(arrayBuffer.slice(0))
  const ch = audioBuf.getChannelData(0)
  const mono = new Float32Array(audioBuf.length)
  mono.set(ch)
  return { samples: mono, sr: audioBuf.sampleRate }
}
