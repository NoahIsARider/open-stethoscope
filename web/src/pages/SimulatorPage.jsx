import React, { useEffect, useRef, useState } from 'react'
import { fetchManifest, POSITIONS, MURMURS, murmurName } from '../api.js'

const LOC_FILTERS = ['全部', ...POSITIONS.map((p) => p.id)]

export default function SimulatorPage() {
  const [manifest, setManifest] = useState(null)
  const [locFilter, setLocFilter] = useState('全部')
  const [murFilter, setMurFilter] = useState('全部')
  const [active, setActive] = useState(null)
  const [playing, setPlaying] = useState(false)
  const [quiz, setQuiz] = useState(false)
  const [guessMur, setGuessMur] = useState(null)
  const [guessLoc, setGuessLoc] = useState(null)
  const [revealed, setRevealed] = useState(false)

  const audioRef = useRef(null)
  const analyserRef = useRef(null)
  const ctxRef = useRef(null)
  const srcRef = useRef(null)
  const rafRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    fetchManifest().then(setManifest).catch(() => {})
  }, [])

  useEffect(() => {
    const draw = () => {
      const cv = canvasRef.current
      if (cv && analyserRef.current) {
        const g = cv.getContext('2d')
        const w = cv.width, h = cv.height
        g.drawImage(cv, -2, 0)
        g.fillStyle = '#0d1b24'
        g.fillRect(w - 2, 0, 2, h)
        const binCount = analyserRef.current.frequencyBinCount
        const data = new Uint8Array(binCount)
        analyserRef.current.getByteFrequencyData(data)
        const n = 96
        for (let i = 0; i < n; i++) {
          const idx = Math.floor(i / n * binCount * 0.35)
          const v = data[idx] / 255
          const hue = 120 - v * 120
          g.fillStyle = `hsl(${hue}, 90%, ${20 + v * 50}%)`
          g.fillRect(w - 2, h - 1 - Math.floor((h / n) * (i + 1)), 2, Math.floor(h / n) + 1)
        }
      }
      rafRef.current = requestAnimationFrame(draw)
    }
    rafRef.current = requestAnimationFrame(draw)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [])

  const play = async (clip) => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.src = ''
      try { srcRef.current?.disconnect() } catch {}
    }
    setActive(clip)
    setPlaying(false)
    setRevealed(false)
    setGuessMur(null)
    setGuessLoc(null)
    if (!ctxRef.current) {
      ctxRef.current = new (window.AudioContext || window.webkitAudioContext)()
    }
    try { await ctxRef.current.resume() } catch {}
    const audio = new Audio(`/api/simulator/audio/${clip.id}`)
    audio.crossOrigin = 'anonymous'
    audioRef.current = audio
    const src = ctxRef.current.createMediaElementSource(audio)
    srcRef.current = src
    const analyser = ctxRef.current.createAnalyser()
    analyser.fftSize = 2048
    src.connect(analyser)
    analyser.connect(ctxRef.current.destination)
    analyserRef.current = analyser
    audio.onended = () => setPlaying(false)
    audio.onplay = () => setPlaying(true)
    audio.onpause = () => setPlaying(false)
    audio.play()
  }

  const togglePlay = () => {
    if (!audioRef.current) return
    if (audioRef.current.paused) audioRef.current.play(); else audioRef.current.pause()
  }

  const clips = manifest?.clips?.filter((c) => {
    if (locFilter !== '全部' && c.location !== locFilter) return false
    if (murFilter !== '全部' && c.murmur !== murFilter) return false
    return true
  }) || []

  const metaFor = (c) => {
    const age = c.age != null ? c.age : '未知'
    const sex = c.sex === 'Female' ? '女' : c.sex === 'Male' ? '男' : '未知'
    return { age, sex }
  }

  return (
    <div>
      <div className="card">
        <h3>听诊教学模拟器 · Auscultation Teaching Simulator</h3>
        <p style={{ margin: '0 0 10px', fontSize: 13, color: 'var(--ink-2)' }}>
          以下为 CirCor DigiScope 数据集中 <b>真实、专家标注</b> 的心音录音（含 4 个标准听诊部位 × 杂音阴性/待定/阳性）。
          逐条听辨：识别第一心音 S1、第二心音 S2 的规律，比较不同部位与杂音的心音特征。
        </p>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          {LOC_FILTERS.map((f) => (
            <button
              key={f}
              className={'quiz-opt' + (locFilter === f ? ' picked' : '')}
              onClick={() => setLocFilter(f)}
            >
              {f === '全部' ? '全部部位' : f}
            </button>
          ))}
          <span style={{ width: 12 }} />
          <button className={'quiz-opt' + (murFilter === '全部' ? ' picked' : '')} onClick={() => setMurFilter('全部')}>
            全部杂音
          </button>
          {MURMURS.map((m) => (
            <button
              key={m.id}
              className={'quiz-opt' + (murFilter === m.id ? ' picked' : '')}
              onClick={() => setMurFilter(m.id)}
            >
              {m.name}
            </button>
          ))}
          <span style={{ width: 12 }} />
          <label className="quiz-opt" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={quiz} onChange={(e) => { setQuiz(e.target.checked); setRevealed(false); setGuessMur(null); setGuessLoc(null) }} />
            练习模式（隐藏答案）
          </label>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>播放器 · Player</h3>
          {!active ? (
            <div className="note">从右侧列表选择一条真实录音开始播放。</div>
          ) : (
            <div>
              <div className="verdict-box ok" style={{ marginTop: 0 }}>
                <div className="check-row" style={{ borderBottom: 'none' }}>
                  <span>
                    录音 <b>{active.id}</b> · 部位 <b>{active.location}</b> · 时长 {active.duration_s}s
                  </span>
                  <span>
                    <button className="btn ghost" style={{ padding: '6px 14px' }} onClick={togglePlay}>
                      {playing ? '暂停' : '播放'}
                    </button>
                  </span>
                </div>
                <div className="check-row" style={{ borderBottom: 'none' }}>
                  <span>受检者</span>
                  <span className="check-value">年龄 {metaFor(active).age} · 性别 {metaFor(active).sex}</span>
                </div>
                <div className="check-row" style={{ borderBottom: 'none' }}>
                  <span>专家标注</span>
                  {quiz && !revealed ? (
                    <span className="chip warn">练习模式：已隐藏答案</span>
                  ) : (
                    <span>
                      <span className="chip neutral">杂音：{murmurName(active.murmur)}</span>{' '}
                      {active.outcome && <span className="chip neutral">结局：{active.outcome}</span>}
                    </span>
                  )}
                </div>
                {active.murmur === 'Present' && active.murmur_detail && !(quiz && !revealed) && (
                  <div className="check-row" style={{ borderBottom: 'none' }}>
                    <span>杂音特征（专家标注）</span>
                    <span className="check-value">
                      {[
                        active.murmur_detail.systolic_timing && `收缩期·${active.murmur_detail.systolic_timing}`,
                        active.murmur_detail.systolic_shape && `形态·${active.murmur_detail.systolic_shape}`,
                        active.murmur_detail.systolic_grading && `分级·${active.murmur_detail.systolic_grading}`,
                        active.murmur_detail.systolic_pitch && `音调·${active.murmur_detail.systolic_pitch}`,
                        active.murmur_detail.systolic_quality && `音质·${active.murmur_detail.systolic_quality}`,
                        active.murmur_detail.diastolic_timing && `舒张期·${active.murmur_detail.diastolic_timing}`,
                      ].filter(Boolean).join(' · ')}
                    </span>
                  </div>
                )}
              </div>
              {quiz && (
                <div>
                  <div className="quiz-form">
                    <b>杂音判断：</b>
                    {MURMURS.map((m) => (
                      <button key={m.id} className={'quiz-opt' + (guessMur === m.id ? ' picked' : '')} onClick={() => setGuessMur(m.id)}>
                        {m.name}
                      </button>
                    ))}
                  </div>
                  <div className="quiz-form">
                    <b>部位判断：</b>
                    {POSITIONS.map((p) => (
                      <button key={p.id} className={'quiz-opt' + (guessLoc === p.id ? ' picked' : '')} onClick={() => setGuessLoc(p.id)}>
                        {p.id}
                      </button>
                    ))}
                  </div>
                  <div style={{ marginTop: 12, display: 'flex', gap: 10, alignItems: 'center' }}>
                    <button className="btn" onClick={() => setRevealed(true)}>揭晓答案</button>
                    {revealed && (
                      <span>
                        {guessMur === active.murmur ? (
                          <span className="chip ok">杂音判断正确</span>
                        ) : (
                          <span className="chip fail">杂音判断有误（应为 {murmurName(active.murmur)}）</span>
                        )}{' '}
                        {guessLoc === active.location ? (
                          <span className="chip ok">部位判断正确</span>
                        ) : (
                          <span className="chip fail">部位判断有误（应为 {active.location}）</span>
                        )}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
          <canvas ref={canvasRef} className="spectrum" width={480} height={220} />
          <div className="note">频谱瀑布图：绿色深者为低频能量集中（心音 S1/S2 集中在 25-150 Hz），随频率升高能量递减。杂音阳性录音常在收缩/舒张期出现高次谐波能量带。</div>
        </div>

        <div className="card">
          <h3>真实录音库 · Recording Library（{clips.length} 条）</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {clips.map((c) => {
              const { age, sex } = metaFor(c)
              const m = MURMURS.find((x) => x.id === c.murmur)
              return (
                <div key={c.id} className={'clip-card' + (active?.id === c.id ? ' active' : '')}>
                  <div className="clip-head">
                    <span>{c.location} · {c.duration_s}s</span>
                    {quiz ? (
                      <span className="chip neutral">练习</span>
                    ) : (
                      <span className="chip neutral" style={{ borderColor: m.color, color: m.color }}>
                        {m.name}
                      </span>
                    )}
                  </div>
                  <div className="clip-body">
                    <div className="row"><span>受检者</span><b>{age}岁 {sex}</b></div>
                    <div className="row"><span>专家标注</span><b>{quiz ? '—' : murmurName(c.murmur)}</b></div>
                    {c.outcome && !quiz && <div className="row"><span>临床结局</span><b>{c.outcome}</b></div>}
                    {c.murmur_detail && !quiz && (
                      <div className="row"><span>杂音特征</span><b>{c.murmur_detail.systolic_timing || '—'}·{c.murmur_detail.systolic_grading || '—'}</b></div>
                    )}
                  </div>
                  <div className="clip-foot">
                    <button className="play-btn" onClick={() => play(c)}>
                      {active?.id === c.id && playing ? '正在播放…' : '播放'}
                    </button>
                  </div>
                </div>
              )
            })}
            {clips.length === 0 && <div className="note">当前筛选条件下无录音。</div>}
          </div>
        </div>
      </div>

      <div className="card">
        <h3>听辨要点 · How to Listen</h3>
        <div className="grid-3">
          <div>
            <h4>S1 第一心音</h4>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
              <li>二尖瓣、三尖瓣关闭产生，标志收缩期开始</li>
              <li>心尖部（MV）最响，较 S2 长而低沉</li>
              <li>听诊时先定 S1/S2，再判断收缩期 / 舒张期</li>
            </ul>
          </div>
          <div>
            <h4>S2 第二心音</h4>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
              <li>主动脉瓣、肺动脉瓣关闭产生，标志舒张期开始</li>
              <li>心底（AV/PV）较响，比 S1 短促、音调高</li>
              <li>“心尖-S1，心底-S2”，先找 S2 再反推 S1</li>
            </ul>
          </div>
          <div>
            <h4>杂音识别</h4>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
              <li>收缩期杂音位于 S1 之后、S2 之前</li>
              <li>舒张期杂音位于 S2 之后、下一个 S1 之前</li>
              <li>杂音阳性录音可闻及高频“吹风样”或“隆隆样”额外成分</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
