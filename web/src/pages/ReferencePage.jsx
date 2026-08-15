import React, { useEffect, useState } from 'react'
import { fetchModelInfo, POSITIONS } from '../api.js'

const QUALITY_REF = [
  { metric: '时长', good: '≥ 8 s', value: '8-12 s（与 CirCor 标准一致）', note: '过短无法覆盖完整心周期' },
  { metric: '信噪比', good: '≥ 12 dB', value: '6-12 dB 可接受；<6 dB 不合格', note: '环境噪声为主因' },
  { metric: '削波', good: '0%', value: '>1% 判定 ADC 饱和', note: '增益过高所致，应降低增益' },
  { metric: '频谱平坦度', good: '< 0.45', value: '≥0.6 判定宽谱噪声主导', note: '平坦度越高越接近白噪声' },
  { metric: '心音节律', good: '60-100 bpm', value: '可检测范围 30-250 bpm', note: '检测不到节律提示接触不良' },
]

export default function ReferencePage() {
  const [info, setInfo] = useState(null)

  useEffect(() => {
    fetchModelInfo().then(setInfo).catch(() => {})
  }, [])

  return (
    <div>
      <div className="card">
        <h3>标准听诊部位 · Auscultation Landmarks</h3>
        <table className="tbl">
          <thead>
            <tr><th>部位</th><th>解剖位置</th><th>主要听诊内容</th></tr>
          </thead>
          <tbody>
            <tr><td><b>AV 主动脉瓣区</b></td><td>胸骨右缘第 2 肋间</td><td>S2 主动脉瓣成分、主动脉瓣杂音</td></tr>
            <tr><td><b>PV 肺动脉瓣区</b></td><td>胸骨左缘第 2 肋间</td><td>S2 肺动脉瓣成分、肺动脉瓣杂音</td></tr>
            <tr><td><b>TV 三尖瓣区</b></td><td>胸骨左缘第 4-5 肋间</td><td>三尖瓣关闭/狭窄杂音、右心事件</td></tr>
            <tr><td><b>MV 二尖瓣区（心尖部）</b></td><td>左锁骨中线第 5 肋间</td><td>S1、二尖瓣病变杂音（最常用）</td></tr>
          </tbody>
        </table>
        <div className="note">
          标准听诊顺序建议：MV → PV → AV → TV（先心尖后心底，再沿胸骨缘移动），每个部位至少听 2 个完整心周期。
        </div>
      </div>

      <div className="card">
        <h3>录音质量标准 · Recording Quality Criteria</h3>
        <table className="tbl">
          <thead>
            <tr><th>指标</th><th>合格标准</th><th>分级阈值</th><th>说明</th></tr>
          </thead>
          <tbody>
            {QUALITY_REF.map((r) => (
              <tr key={r.metric}>
                <td>{r.metric}</td>
                <td><b>{r.good}</b></td>
                <td>{r.value}</td>
                <td>{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="note">
          参考：CirCor DigiScope 采集管线采用 25-400 Hz 带通（Oliveira et al., 2021）；心音能量主要集中在 25-150 Hz。
        </div>
      </div>

      <div className="card">
        <h3>已部署模型 · Deployed Models（持出测试评估）</h3>
        {!info ? (
          <div className="note">后端未返回模型信息。</div>
        ) : (
          <div className="grid-2">
            <div>
              <h4>听诊位置分类（{info.position_model.classes.join(' / ')}）</h4>
              <div className="stat-row" style={{ marginBottom: 10 }}>
                <div className="stat-box"><div className="v">{(info.position_model.test * 100).toFixed(1)}%</div><div className="l">持出测试准确率</div></div>
                <div className="stat-box"><div className="v">{info.position_model.test_macro_f1.toFixed(3)}</div><div className="l">宏平均 F1</div></div>
              </div>
              <div className="note">
                训练数据：{info.position_model.trained_on}。划分：{info.position_model.split}。
                混淆矩阵：{JSON.stringify(info.position_model.confusion_matrix)}
              </div>
            </div>
            <div>
              <h4>杂音筛查（{info.murmur_screening.classes.join(' / ')}）</h4>
              <div className="stat-row" style={{ marginBottom: 10 }}>
                <div className="stat-box"><div className="v">{(info.murmur_screening.test_accuracy * 100).toFixed(1)}%</div><div className="l">持出测试准确率</div></div>
                <div className="stat-box"><div className="v">{info.murmur_screening.test_macro_f1.toFixed(3)}</div><div className="l">宏平均 F1</div></div>
                <div className="stat-box"><div className="v">{info.murmur_screening.test_s_murmur.toFixed(3)}</div><div className="l">s_murmur（官方指标）</div></div>
              </div>
              <div className="note">{info.murmur_screening.disclaimer}</div>
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h3>录音规范建议 · Recording Protocol</h3>
        <div className="guide-step">
          <div className="guide-num">1</div>
          <div className="guide-body">
            <b>环境准备</b>
            <span>关闭空调、风扇，远离气流与人群；嘱患者保持安静、平稳呼吸（必要时屏气数秒）；手机/听诊器远离衣物摩擦。</span>
          </div>
        </div>
        <div className="guide-step">
          <div className="guide-num">2</div>
          <div className="guide-body">
            <b>胸件放置</b>
            <span>选用钟形胸件（低频杂音更清楚）或膜型胸件（高频更清楚）；垂直于皮肤，完全贴合，避免按压过紧产生摩擦音。</span>
          </div>
        </div>
        <div className="guide-step">
          <div className="guide-num">3</div>
          <div className="guide-body">
            <b>录音执行</b>
            <span>每部位连续录音 ≥8 秒；如信号削波（红灯），降低增益；如噪声过大（红灯），先排查环境噪声。</span>
          </div>
        </div>
        <div className="guide-step">
          <div className="guide-num">4</div>
          <div className="guide-body">
            <b>数据归档</b>
            <span>质检合格后归档；不合格录音应重新采集，避免污染训练/分析数据（本工具存在的意义）。</span>
          </div>
        </div>
      </div>
    </div>
  )
}
