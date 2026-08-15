import React, { useCallback, useEffect, useRef, useState } from 'react'
import { analyzePcm, POSITIONS, MURMURS } from '../api.js'

const MAX_BUF_S = 15
const QC_WINDOW_S = 12
const LIVE_INTERVAL_MS = 2000

export default function RecorderPage() {
  const [declared, setDeclared] = useState(null)
  const [recording, setRecording] = useState(false)
  const [live, setLive] = useState(null)
  const [final, setFinal] = useState(null)
  const [err, setErr] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)

  const accRef = useRef(new Float32Array(0))
  const audioRef = useRef(null)
  const jsRef = useRef(null)
  const ctxRef = useRef(null)
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const rafRef = useRef(null)
  const liveRef = useRef(null)
  const canvasRef = useRef(null)

  const append = useCallback((chunk) => {
    const prev = accRef.current
    const next = new Float32Array(prev.length + chunk.length)
    next.set(prev, 0)
    next.set(chunk, prev.length)
    const cap = MAX_BUF_S * ctxRef.current.sampleRate
    if (next.length > cap) {
      accRef.current = next.subarray(next.length - cap)
    } else {
      accRef.current = next
    }
  }, [])

  const start = useCallback(async () => {
    setErr(null)
    setFinal(null)
    setLive(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      })
      streamRef.current = stream
      const ctx = new AudioContext()
      ctxRef.current = ctx
      await ctx.resume()
      const src = ctx.createMediaStreamSource(stream)
      const js = ctx.createScriptProcessor(4096, 1, 1)
      jsRef.current = js
      js.onaudioprocess = (e) => {
        const ch = e.inputBuffer.getChannelData(0)
        const copy = new Float32Array(ch.length)
        copy.set(ch)
        append(copy)
      }
      src.connect(js)
      setRecording(true)    } catch (e) {
      setErr('无法获取麦克风权限。请确认浏览器允许使用麦克风，并确认已接入数字听诊器或外接麦克风。' + e.message)
    }
  }, [append])

  const stop = useCallback(async () => {
    setRecording(false)
    if (timerRef.current) clearInterval(timerRef.current)
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    try { jsRef.current?.disconnect() } catch {}
    const ctx = ctxRef.current
    const sr = ctx ? ctx.sampleRate : 4000
    try { ctx?.close() } catch {}
    try { streamRef.current?.getTracks().forEach((t) => t.stop()) } catch {}
    ctxRef.current = null
    if (accRef.current.length > 0) {
      setAnalyzing(true)
      try {
        const res = await analyzePcm(accRef.current, sr, declared)
        setFinal(res)
      } catch (e) {
        setErr('质检分析失败：' + e.message)
      }
      setAnalyzing(false)
    }
  }, [declared])

  useEffect(() => {
    if (recording) {
      timerRef.current = setInterval(async () => {
        if (!ctxRef.current || accRef.current.length === 0) return
        const tail = accRef.current.subarray(Math.max(0, accRef.current.length - QC_WINDOW_S * ctxRef.current.sampleRate))
        try {
          const res = await analyzePcm(tail, ctxRef.current.sampleRate, declared)
          liveRef.current = res
          setLive(res)
        } catch {}
      }, LIVE_INTERVAL_MS)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [recording, declared])

  useEffect(() => {
    const draw = () => {
      const cv = canvasRef.current
      if (cv && ctxRef.current) {
        const x = accRef.current
        const w = cv.width, h = cv.height
        const g = cv.getContext('2d')
        g.clearRect(0, 0, w, h)
        g.fillStyle = '#0d1b24'
        g.fillRect(0, 0, w, h)
        g.strokeStyle = '#4fd1a5'
        g.lineWidth = 1.2
        g.beginPath()
        const n = x.length
        if (n > 0) {
          const step = Math.max(1, Math.floor(n / (w * 2)))
          for (let i = 0; i < w; i++) {
            const s = Math.min(n - 1, i * step)
            const v = x[s]
            const y = h / 2 - v * h * 0.42
            if (i === 0) g.moveTo(i, y); else g.lineTo(i, y)
          }
          g.stroke()
        }
      }
      rafRef.current = requestAnimationFrame(draw)
    }
    rafRef.current = requestAnimationFrame(draw)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [])

  const qc = live || final
  const statusColor = (s) => ({ ok: 'ok', warn: 'warn', fail: 'fail' }[s] || 'neutral')

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setErr(null)
    setFinal(null)
    setLive(null)
    setAnalyzing(true)
    try {
      const buf = await file.arrayBuffer()
      const ac = new (window.AudioContext || window.webkitAudioContext)()
      const audio = await ac.decodeAudioData(buf)
      const ch = audio.getChannelData(0)
      const res = await analyzePcm(ch, audio.sampleRate, declared)
      setFinal(res)
      setLive(res)
    } catch (err2) {
      setErr('音频解析失败：' + err2.message + '（请上传 WAV/音频文件）')
    }
    setAnalyzing(false)
    e.target.value = ''
  }

  const posVerdict = qc?.position
  const declaredInfo = POSITIONS.find((p) => p.id === declared)

  return (
    <div>
      <div className="card">
        <h3>录音规范化引导 · Standardized Recording Protocol</h3>
        <div className="guide-step">
          <div className="guide-num">1</div>
          <div className="guide-body">
            <b>选择听诊部位</b>
            <span>四选一：主动脉瓣区（AV）、肺动脉瓣区（PV）、三尖瓣区（TV）、二尖瓣区/心尖部（MV）</span>
          </div>
        </div>
        <div className="pos-grid">
          {POSITIONS.map((p) => (
            <div
              key={p.id}
              className={'pos-opt' + (declared === p.id ? ' selected' : '')}
              onClick={() => setDeclared(p.id)}
            >
              <div className="pos-code">{p.id}</div>
              <div className="pos-name">{p.name}</div>
              <div className="pos-point">{p.point}</div>
            </div>
          ))}
        </div>

        <div className="guide-step" style={{ marginTop: 12 }}>
          <div className="guide-num">2</div>
          <div className="guide-body">
            <b>放置胸件并开始录音</b>
            <span>胸件应垂直于皮肤并保持完全贴合，减少摩擦；要求环境安静，避免说话、衣物摩擦、空调气流；连续录音 ≥8 秒</span>
          </div>
        </div>
        <div className="guide-step">
          <div className="guide-num">3</div>
          <div className="guide-body">
            <b>查看实时质检结果</b>
            <span>系统实时评估：信号电平、削波、信噪比、频谱平坦度、心音节律、听诊位置一致性</span>
          </div>
        </div>
        <div className="guide-step">
          <div className="guide-num">4</div>
          <div className="guide-body">
            <b>停止并生成录音报告</b>
            <span>质量不合格时按建议调整增益、体位或环境后重新录音</span>
          </div>
        </div>

        <div style={{ marginTop: 16, display: 'flex', gap: 10, alignItems: 'center' }}>
          {!recording ? (
            <button className="btn" onClick={start} disabled={!declared}>
              开始录音
            </button>
          ) : (
            <button className="btn danger" onClick={stop}>
              停止录音
            </button>
          )}
          {recording && (
            <span className="chip fail">
              <span className="rec-ind" /> 录音中
            </span>
          )}
          {analyzing && <span className="chip neutral">正在生成最终报告…</span>}
          {!declared && !recording && (
            <span className="chip warn">请先选择听诊部位</span>
          )}
        </div>
        <div style={{ marginTop: 12 }}>
          <label className="btn ghost" style={{ display: 'inline-block', cursor: 'pointer' }}>
            上传 WAV 录音文件进行质检
            <input type="file" accept="audio/*,.wav" onChange={handleUpload} style={{ display: 'none' }} />
          </label>
          <span className="note" style={{ marginTop: 8, display: 'block' }}>
            无麦克风环境亦可使用：上传真实 WAV 录音（如 CirCor 样本）即可得到完整质检报告。
          </span>
        </div>
        {err && <div className="verdict-box fail">{err}</div>}
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>实时波形 · Live Waveform</h3>
          <canvas ref={canvasRef} className="waveform" width={640} height={90} />
        </div>

        <div className="card">
          <h3>实时质控指标 · Live Metrics</h3>
          {!qc ? (
            <div className="note">开始录音后，系统将每 2 秒更新一次实时质控指标。</div>
          ) : (
            <div>
              <div className="stat-row">
                <div className="stat-box">
                  <div className="v">{qc.duration_s.toFixed(1)}s</div>
                  <div className="l">时长</div>
                </div>
                <div className="stat-box">
                  <div className="v">{qc.snr_db != null ? qc.snr_db.toFixed(1) : '—'}</div>
                  <div className="l">信噪比 dB</div>
                </div>
                <div className="stat-box">
                  <div className="v">{qc.heart_rate_bpm != null ? qc.heart_rate_bpm.toFixed(0) : '—'}</div>
                  <div className="l">心率 bpm</div>
                </div>
                <div className="stat-box">
                  <div className="v">{qc.noise_floor_db != null ? qc.noise_floor_db.toFixed(0) : '—'}</div>
                  <div className="l">噪声底 dB</div>
                </div>
              </div>
              <div style={{ marginTop: 12 }}>
                {qc.checks.map((c) => (
                  <div className="check-row" key={c.metric}>
                    <span className="check-name">
                      <span className={'chip ' + statusColor(c.status)}>
                        {{ ok: '合格', warn: '警告', fail: '不合格' }[c.status]}
                      </span>{' '}
                      {c.label}
                    </span>
                    <span className="check-value">{c.value != null ? c.value : ''}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>听诊位置校验 · Position Verification</h3>
          {!posVerdict ? (
            <div className="note">录音数据不足，尚无法校验听诊位置。</div>
          ) : (
            <div>
              <div className={'verdict-box ' + posVerdict.status}>
                <b>
                  {posVerdict.status === 'ok' && '位置校验通过：'}
                  {posVerdict.status === 'warn' && '位置置信度不足，请核对放置：'}
                  {posVerdict.status === 'fail' && '无法确认听诊位置，请重新放置胸件：'}
                </b>
                <div style={{ marginTop: 6 }}>
                  模型识别部位：<b>{posVerdict.top_location}</b>{' '}
                  （置信度 {(posVerdict.confidence * 100).toFixed(0)}%）
                  {posVerdict.declared && (
                    <span>
                      {' '}· 目标部位：<b>{posVerdict.declared}</b>
                      {posVerdict.match ? ' ✓ 一致' : ' ✗ 不一致'}
                    </span>
                  )}
                </div>
                {posVerdict.declared && !posVerdict.match && (
                  <div className="recommend">
                    请按标准听诊点重新放置胸件：
                    {declaredInfo && (
                      <b> {declaredInfo.name}（{declaredInfo.point}）</b>
                    )}
                  </div>
                )}
              </div>
              <div style={{ marginTop: 12 }}>
                <h4>部位概率分布</h4>
                {Object.entries(posVerdict.probabilities).map(([loc, p]) => (
                  <div className="prob-bar" key={loc}>
                    <div className="prob-bar-label">
                      <span>{loc}</span>
                      <span>{(p * 100).toFixed(0)}%</span>
                    </div>
                    <div className="prob-bar-track">
                      <div className="prob-bar-fill" style={{ width: `${p * 100}%`, background: 'var(--accent)' }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <h3>信号有效性（筛查提示）· Signal Screening</h3>
          {!qc?.murmur ? (
            <div className="note">录音数据不足。</div>
          ) : (
            <div>
              <div className="note" style={{ marginTop: 0 }}>
                {qc.murmur.note} 当前信号优势判断：<b>{MURMURS.find((m) => m.id === qc.murmur.top)?.name}</b>
              </div>
              {qc.murmur.probabilities && Object.entries(qc.murmur.probabilities).map(([id, p]) => {
                const m = MURMURS.find((x) => x.id === id)
                return (
                  <div className="prob-bar" key={id}>
                    <div className="prob-bar-label">
                      <span>{m?.name}</span>
                      <span>{(p * 100).toFixed(0)}%</span>
                    </div>
                    <div className="prob-bar-track">
                      <div className="prob-bar-fill" style={{ width: `${p * 100}%`, background: m?.color }} />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {final && (
        <div className="card">
          <h3>录音质检报告 · Recording QC Report</h3>
          <table className="tbl">
            <thead>
              <tr>
                <th>指标</th>
                <th>数值</th>
                <th>判定</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody>
              {final.checks.map((c) => (
                <tr key={c.metric}>
                  <td>{c.metric}</td>
                  <td className="num">{c.value != null ? c.value : '—'}</td>
                  <td><span className={'chip ' + statusColor(c.status)}>{{ ok: '合格', warn: '警告', fail: '不合格' }[c.status]}</span></td>
                  <td>{c.label}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="recommend">
            <ul>
              {final.checks.filter((c) => c.status === 'fail').map((c) => (
                <li key={c.metric}>不合格：{c.label}</li>
              ))}
              {final.checks.filter((c) => c.status === 'warn').map((c) => (
                <li key={c.metric}>警告：{c.label}</li>
              ))}
              {final.checks.every((c) => c.status === 'ok') && (
                <li>全部指标合格，本段录音可用于后续心音分析。</li>
              )}
              {final.checks.some((c) => c.status !== 'ok') && (
                <li>请按以上建议调整后重新录音；不合格录音不应进入分析/训练数据集。</li>
              )}
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
