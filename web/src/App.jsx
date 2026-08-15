import React, { useEffect, useState } from 'react'
import RecorderPage from './pages/RecorderPage.jsx'
import SimulatorPage from './pages/SimulatorPage.jsx'
import ReferencePage from './pages/ReferencePage.jsx'
import { fetchModelInfo } from './api.js'

const TABS = [
  { id: 'rec', label: '录音质检', sub: 'Recording QC' },
  { id: 'sim', label: '听诊教学模拟器', sub: 'Teaching Simulator' },
  { id: 'ref', label: '临床参考', sub: 'Reference' },
]

export default function App() {
  const [tab, setTab] = useState('rec')
  const [info, setInfo] = useState(null)

  useEffect(() => {
    fetchModelInfo().then(setInfo).catch(() => {})
  }, [])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">OS</div>
          <div>
            <div className="brand-title">Open Stethoscope · 录音质检与听诊教学</div>
            <div className="brand-sub">
              数字听诊录音规范化质控系统 · 基于 CirCor DigiScope 真实心音数据集
            </div>
          </div>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={'tab' + (tab === t.id ? ' active' : '')}
              onClick={() => setTab(t.id)}
            >
              <span className="tab-label">{t.label}</span>
              <span className="tab-sub">{t.sub}</span>
            </button>
          ))}
        </nav>
      </header>

      {tab === 'rec' && <RecorderPage />}
      {tab === 'sim' && <SimulatorPage />}
      {tab === 'ref' && <ReferencePage />}

      <footer className="footer">
        <span>数据来源：CirCor DigiScope 1.0.3（PhysioNet Challenge 2022），专家标注，真实录音</span>
        <span>质控算法基于真实 DSP 信号处理；模型为持出测试评估，仅供质控与教学，不构成诊断</span>
      </footer>
    </div>
  )
}
