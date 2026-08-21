// main.js — Open Stethoscope in-browser demo logic.
import { melSpectrogram, toFrames, buildInput, resampleTo, loadParams } from './mel.js'

const $ = (id) => document.getElementById(id)
const CLASSES = ['Absent', 'Unknown', 'Present']
const COLORS = ['#1e9e5a', '#e6a23c', '#d64545']
const LOCS = ['AV', 'PV', 'TV', 'MV']

const state = { params: null, session: null, audio: null, sr: 0, ready: false }

// ─── model / params loading ────────────────────────────────────────────────
async function loadModel() {
  setStatus('loading model…', '')
  try {
    const [paramsJson, modelBuf] = await Promise.all([
      fetch('./assets/mel_params.json').then((r) => r.json()),
      fetch('./assets/model.onnx').then((r) => r.arrayBuffer()),
    ])
    state.params = loadParams(paramsJson)

    if (window.ort) ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/'
    state.session = await ort.InferenceSession.create(modelBuf, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
    state.ready = true
    $('analyze').disabled = false
    setStatus('model ready — 404K params, runs locally', 'ok')
  } catch (e) {
    setStatus('model failed to load: ' + e.message, 'err')
    console.error(e)
  }
}

function setStatus(text, cls) {
  $('status-text').textContent = text
  $('dot').className = 'dot' + (cls ? ' ' + cls : '')
}

// ─── audio input ───────────────────────────────────────────────────────────
async function decodeFile(file) {
  const buf = await file.arrayBuffer()
  const ctx = new (window.AudioContext || window.webkitAudioContext)()
  const audio = await ctx.decodeAudioData(buf)
  const samples = audio.getChannelData(0).slice()
  ctx.close()
  return { samples: new Float32Array(samples), sr: audio.sampleRate }
}

async function loadSample(name) {
  const resp = await fetch('./assets/samples/' + name)
  const buf = await resp.arrayBuffer()
  const ctx = new (window.AudioContext || window.webkitAudioContext)()
  const audio = await ctx.decodeAudioData(buf)
  const samples = new Float32Array(audio.getChannelData(0))
  ctx.close()
  return { samples, sr: audio.sampleRate }
}

// ─── analysis ──────────────────────────────────────────────────────────────
async function analyze() {
  if (!state.ready || !state.audio) return
  const p = state.params
  const x = resampleTo(state.audio.samples, state.audio.sr, p.sr)
  const locIdx = LOCS.indexOf($('loc').value)
  if (locIdx < 0) return

  const t0 = performance.now()
  const mel = toFrames(melSpectrogram(x, p), p.n_frames, p.n_mels)
  const { xb, mask } = buildInput(mel, locIdx, p)
  const feeds = {
    x: new ort.Tensor('float32', xb, [1, 4, 1, p.n_mels, p.n_frames]),
    mask: new ort.Tensor('bool', mask, [1, 4]),
  }
  const out = await state.session.run(feeds)
  const logits = out.logits.data
  const max = Math.max(...logits)
  const exps = logits.map((v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map((v) => v / sum)
  const ms = (performance.now() - t0).toFixed(0)

  renderResults(probs, mel, x, ms)
}

function renderResults(probs, mel, x, ms) {
  const top = probs.indexOf(Math.max(...probs))
  $('verdict-label').textContent = CLASSES[top]
  $('verdict-label').style.color = COLORS[top]
  $('verdict-conf').textContent =
    (probs[top] * 100).toFixed(1) + '% · inference ' + ms + ' ms (local, WASM)'
  for (let i = 0; i < 3; i++) {
    $('fill-' + i).style.width = (probs[i] * 100).toFixed(1) + '%'
    $('pct-' + i).textContent = (probs[i] * 100).toFixed(1) + '%'
  }
  drawWave(x)
  drawMel(mel, state.params)
}

function drawWave(x) {
  const c = $('wave'), ctx = c.getContext('2d')
  const W = c.width, H = c.height
  ctx.clearRect(0, 0, W, H)
  ctx.fillStyle = '#0d1f1b'
  ctx.fillRect(0, 0, W, H)
  ctx.strokeStyle = '#38d9a9'
  ctx.lineWidth = 1
  const step = Math.max(1, Math.floor(x.length / W))
  ctx.beginPath()
  for (let px = 0; px < W; px++) {
    const v = x[px * step]
    const y = H / 2 - v * H * 0.42
    px === 0 ? ctx.moveTo(px, y) : ctx.lineTo(px, y)
  }
  ctx.stroke()
  ctx.strokeStyle = 'rgba(56,217,169,.15)'
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke()
}

function drawMel(mel, p) {
  const c = $('mel'), ctx = c.getContext('2d')
  const W = c.width, H = c.height
  const nMels = p.n_mels, nFrames = p.n_frames
  const img = ctx.createImageData(W, H)
  const bw = W / nFrames, bh = H / nMels
  for (let px = 0; px < W; px++) {
    const t = Math.min(nFrames - 1, Math.floor(px / bw))
    for (let py = 0; py < H; py++) {
      const m = nMels - 1 - Math.min(nMels - 1, Math.floor(py / bh))
      const v = mel[m * nFrames + t]
      const n = Math.min(1, Math.max(0, (v + 80) / 80))
      const i = (py * W + px) * 4
      img.data[i] = Math.round(12 + n * 40)      // R
      img.data[i + 1] = Math.round(40 + n * 180) // G
      img.data[i + 2] = Math.round(40 + n * 120) // B
      img.data[i + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
}

// ─── events ────────────────────────────────────────────────────────────────
document.querySelectorAll('#samples .btn').forEach((btn) => {
  btn.addEventListener('click', async () => {
    btn.disabled = true
    setStatus('loading ' + btn.dataset.sample + '…', '')
    try {
      const audio = await loadSample(btn.dataset.sample)
      state.audio = audio
      $('loc').value = 'AV' // sample clips are aortic-position recordings
      setStatus('loaded ' + btn.dataset.sample + ' (' + (audio.sr / 1000).toFixed(1) + ' kHz, ' + (audio.samples.length / audio.sr).toFixed(1) + ' s)', 'ok')
      analyze()
    } catch (e) {
      setStatus('failed to load sample: ' + e.message, 'err')
    } finally {
      btn.disabled = false
    }
  })
})

$('file').addEventListener('change', async (e) => {
  const file = e.target.files[0]
  if (!file) return
  setStatus('decoding ' + file.name + '…', '')
  try {
    const audio = await decodeFile(file)
    state.audio = audio
    setStatus('loaded ' + file.name + ' (' + (audio.sr / 1000).toFixed(1) + ' kHz, ' + (audio.samples.length / audio.sr).toFixed(1) + ' s)', 'ok')
    analyze()
  } catch (err) {
    setStatus('failed to decode audio: ' + err.message, 'err')
  }
})

$('analyze').addEventListener('click', analyze)

loadModel()
