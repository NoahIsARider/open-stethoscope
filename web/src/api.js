export async function analyzePcm(float32, sr, declared) {
  const buf = new ArrayBuffer(float32.length * 4)
  new Float32Array(buf).set(float32)
  const resp = await fetch('/api/qc/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/octet-stream',
      'X-Sample-Rate': String(sr),
      ...(declared ? { 'X-Declared-Position': declared } : {}),
    },
    body: buf,
  })
  if (!resp.ok) throw new Error('analyze failed: ' + resp.status)
  return resp.json()
}

export async function fetchManifest() {
  const resp = await fetch('/api/simulator/manifest')
  return resp.json()
}

export async function fetchModelInfo() {
  const resp = await fetch('/api/models/info')
  return resp.json()
}

export const POSITIONS = [
  { id: 'AV', name: '主动脉瓣区', point: '胸骨右缘第 2 肋间' },
  { id: 'PV', name: '肺动脉瓣区', point: '胸骨左缘第 2 肋间' },
  { id: 'TV', name: '三尖瓣区', point: '胸骨左缘第 4-5 肋间' },
  { id: 'MV', name: '二尖瓣区（心尖部）', point: '左锁骨中线第 5 肋间' },
]

export const MURMURS = [
  { id: 'Absent', name: '杂音阴性', color: '#1e9e5a' },
  { id: 'Unknown', name: '杂音待定', color: '#e6a23c' },
  { id: 'Present', name: '杂音阳性', color: '#d64545' },
]

export function murmurName(id) {
  const m = MURMURS.find((x) => x.id === id)
  return m ? m.name : id
}
