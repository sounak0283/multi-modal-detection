import { useCallback, useEffect, useRef, useState } from 'react'
import { Badge, Button, Card, CardBody, CardHead, EmptyState, Field, Select } from '../components/ui'
import { useToast } from '../components/Toast'
import { usePolling } from '../lib/usePolling'
import Boundaries from './Boundaries'
import { api, testStreamUrl, testVideoFrameUrl, uploadTestVideo } from '../api'

const fmtTime = (s) => {
  const total = Math.max(0, Math.round(s || 0))
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}
const fmtSize = (bytes) =>
  bytes > 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`

const SEVERITY_TONE = { critical: 'alarm', high: 'alarm', medium: 'warn', low: 'neutral' }

function ModulePicker({ modules, value, onChange }) {
  const byId = Object.fromEntries(modules.map((m) => [m.id, m]))

  const toggle = (id) => {
    const on = value.includes(id)
    let next = on ? value.filter((v) => v !== id) : [...value, id]
    if (!on && byId[id]?.requires_boundary && !next.includes('boundary')) next = [...next, 'boundary']
    if (on && id === 'boundary') {
      next = next.filter((v) => !byId[v]?.requires_boundary)
    }
    onChange(next)
  }

  return (
    <div className="flex flex-wrap gap-2">
      {modules.map((m) => {
        const on = value.includes(m.id)
        return (
          <button
            key={m.id}
            type="button"
            disabled={!m.available}
            title={m.available ? '' : m.reason}
            onClick={() => toggle(m.id)}
            className={[
              'rounded-lg border px-3 py-1.5 text-[12.5px] transition-colors',
              on
                ? 'border-brand-500 bg-brand-500/15 text-ink-50'
                : 'border-ink-700 bg-ink-800 text-ink-300 hover:border-ink-600',
              m.available ? '' : 'cursor-not-allowed opacity-40',
            ].join(' ')}
          >
            {m.label}
          </button>
        )
      })}
    </div>
  )
}

function Setup({ status, onStarted }) {
  const toast = useToast()
  const fileRef = useRef(null)
  const [progress, setProgress] = useState(null)
  const [videoId, setVideoId] = useState('')
  const [modules, setModules] = useState(['fire_smoke'])
  const [zonesFrom, setZonesFrom] = useState('')
  const [loop, setLoop] = useState(false)
  const [sendAlerts, setSendAlerts] = useState(false)
  const [starting, setStarting] = useState(false)
  const [zones, setZones] = useState([])
  const [selected, setSelected] = useState(-1)

  const uploads = status.uploads ?? []
  const needsZones = modules.includes('boundary')

  useEffect(() => {
    if (!videoId && uploads.length) setVideoId(uploads[0].id)
  }, [uploads, videoId])

  const onFile = async (event) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    if (file.size > status.max_upload_mb * 1048576) {
      toast(`That file is larger than the ${status.max_upload_mb} MB limit.`, 'error')
      return
    }
    setProgress(0)
    try {
      const saved = await uploadTestVideo(file, setProgress)
      setVideoId(saved.id)
      toast(`Uploaded ${saved.name}.`, 'ok')
      await onStarted(null)
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setProgress(null)
    }
  }

  const remove = async (id) => {
    try {
      await api.deleteTestVideo(id)
      if (id === videoId) setVideoId('')
      await onStarted(null)
    } catch (err) {
      toast(err.message, 'error')
    }
  }

  const copyZones = async (cameraId) => {
    setZonesFrom(cameraId)
    if (!cameraId) return
    try {
      const body = await api.getZones(cameraId)
      setZones(body.zones ?? [])
      setSelected(-1)
      toast(`Copied ${body.zones?.length ?? 0} boundaries. Edit them here; the camera is not changed.`, 'ok')
    } catch (err) {
      toast(err.message, 'error')
    }
  }

  const start = async () => {
    setStarting(true)
    try {
      await api.startTestSession({
        video_id: videoId,
        modules,
        zones: needsZones ? zones : [],
        loop,
        send_alerts: sendAlerts,
      })
      await onStarted(null)
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHead title="1. Video" />
        <CardBody className="space-y-3">
          <input ref={fileRef} type="file" accept="video/*,.mkv,.avi,.mov,.mp4,.webm,.m4v" hidden onChange={onFile} />
          <Button variant="primary" disabled={progress !== null} onClick={() => fileRef.current?.click()}>
            {progress === null ? 'Upload a video' : `Uploading ${Math.round(progress * 100)}%`}
          </Button>
          <p className="text-[12px] text-ink-400">
            mp4, avi, mov, mkv, webm or m4v, up to {status.max_upload_mb} MB. Files are deleted after{' '}
            {status.retention_days} days.
          </p>
          {uploads.length === 0 ? (
            <EmptyState>No videos yet.</EmptyState>
          ) : (
            <ul className="divide-y divide-ink-800 rounded-lg border border-ink-800">
              {uploads.map((v) => (
                <li key={v.id} className="flex items-center gap-3 px-3 py-2">
                  <label className="flex flex-1 cursor-pointer items-center gap-3 text-[13px] text-ink-100">
                    <input type="radio" name="video" checked={videoId === v.id} onChange={() => setVideoId(v.id)} />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate">{v.name}</span>
                      <span className="text-[11.5px] text-ink-400">
                        {fmtTime(v.duration_s)} · {v.width}×{v.height} · {Math.round(v.fps)} fps · {fmtSize(v.size_bytes)}
                      </span>
                    </span>
                  </label>
                  <Button variant="ghost" onClick={() => remove(v.id)}>
                    Delete
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHead title="2. Models" />
        <CardBody className="space-y-4">
          <ModulePicker modules={status.modules} value={modules} onChange={setModules} />
          <p className="text-[12px] text-ink-400">
            Choose one or several. Fire/smoke works on its own, with no boundary. Crowd, PPE and
            face recognition need the boundary (person detection), so it is added for you.
          </p>
          {needsZones && (
            <Field
              label="Start from a camera's boundaries (optional)"
              hint="Copies them into the editor below. Your edits stay in this test only."
            >
              <Select value={zonesFrom} onChange={(e) => copyZones(e.target.value)}>
                <option value="">None - draw my own</option>
                {(status.cameras ?? []).map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name || c.id} ({c.zones?.length ?? 0} boundaries)
                  </option>
                ))}
              </Select>
            </Field>
          )}
          <label className="flex items-center gap-2 text-[13px] text-ink-200">
            <input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} />
            Loop the video
          </label>
          <label className="flex items-start gap-2 text-[13px] text-ink-200">
            <input className="mt-1" type="checkbox" checked={sendAlerts} onChange={(e) => setSendAlerts(e.target.checked)} />
            <span>
              Send as real alerts
              <span className="block text-[12px] text-ink-400">
                Saves to alert history, records clips and sends e-mail, labelled [TEST VIDEO]. Off
                by default.
              </span>
            </span>
          </label>
          <Button variant="primary" disabled={!videoId || modules.length === 0 || starting} onClick={start}>
            {starting ? 'Starting…' : 'Run test'}
          </Button>
        </CardBody>
      </Card>
      {needsZones && videoId && (
        <div className="lg:col-span-2">
          <p className="mb-2 text-[12.5px] text-ink-300">
            3. Draw boundaries on a frame of your video. Entry alerts fire for people who walk in
            during the video, not for anyone already inside at the start.
          </p>
          <Boundaries
            key={videoId}
            backdropSrc={testVideoFrameUrl(videoId)}
            zones={zones}
            setZones={setZones}
            selected={selected}
            setSelected={setSelected}
            markDirty={() => {}}
          />
        </div>
      )}
    </div>
  )
}

function Running({ session, onStop, onAgain }) {
  const running = session.status === 'running'
  const tone = running ? 'ok' : 'neutral'
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      <Card>
        <CardHead title={session.video?.name ?? 'Test video'}>
          <Badge tone={tone}>{session.status}</Badge>
        </CardHead>
        <div className="bg-black">
          <img src={testStreamUrl(session.id)} alt="Test video with detections" className="mx-auto max-h-[70vh] w-full object-contain" />
        </div>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-1 px-4 py-3 text-[12.5px] text-ink-300">
          <span>Time {fmtTime(session.elapsed_s)} / {fmtTime(session.video?.duration_s)}</span>
          <span>{session.stats.frames} frames</span>
          <span>{session.stats.detections} detections</span>
          <span>{session.stats.people} people</span>
          <span>{session.stats.inference_ms} ms</span>
          <span className="ml-auto flex gap-2">
            {running ? <Button variant="danger" onClick={onStop}>Stop</Button> : <Button variant="primary" onClick={onAgain}>Run another test</Button>}
          </span>
        </div>
        <div className="border-t border-ink-800 px-4 py-2 text-[12px] text-ink-400">
          Models: {session.modules.join(', ')}
          {session.send_alerts ? ' · alerts are also sent for real' : ''}
        </div>
      </Card>

      <Card>
        <CardHead title={`Alerts (${session.alert_count})`} />
        {session.allAlerts.length === 0 ? (
          <EmptyState>{running ? 'Nothing raised yet.' : 'No alerts were raised.'}</EmptyState>
        ) : (
          <ul className="max-h-[70vh] divide-y divide-ink-800 overflow-y-auto">
            {[...session.allAlerts].reverse().map((a) => (
              <li key={a.id} className="px-4 py-2.5">
                <div className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono text-ink-400">{fmtTime(a.video_time_s)}</span>
                  <Badge tone={SEVERITY_TONE[a.severity] ?? 'neutral'}>{a.kind}</Badge>
                  {a.subtype && <span className="text-ink-400">{a.subtype}</span>}
                </div>
                <p className="mt-1 text-[13px] text-ink-100">{a.message}</p>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  )
}

export default function VideoTest() {
  const toast = useToast()
  const [status, setStatus] = useState(null)
  const refresh = useCallback(async () => {
    try {
      setStatus(await api.testVideoStatus())
    } catch (err) {
      toast(err.message, 'error')
    }
  }, [toast])
  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 4000)
    return () => clearInterval(id)
  }, [refresh])
  const [session, setSession] = useState(null)
  const shown = useRef(0)
  const alertsRef = useRef([])
  const hasSession = Boolean(status?.session)

  const poll = useCallback(async () => {
    try {
      const body = await api.getTestSession(shown.current)
      const s = body.session
      if (!s) {
        setSession(null)
        return
      }
      if (alertsRef.current.length > 0 && s.id !== alertsRef.current[0]?.sessionId) {
        alertsRef.current = []
        shown.current = 0
      }
      const fresh = s.alerts.map((a) => ({ ...a, sessionId: s.id }))
      alertsRef.current = [...alertsRef.current, ...fresh]
      shown.current = alertsRef.current.length
      setSession({ ...s, allAlerts: alertsRef.current })
    } catch (err) {
      toast(err.message, 'error')
    }
  }, [toast])

  usePolling(poll, 1500, hasSession)

  const refreshAll = async () => {
    alertsRef.current = []
    shown.current = 0
    setSession(null)
    await refresh()
  }

  const stop = async () => {
    try {
      await api.stopTestSession()
      await refreshAll()
    } catch (err) {
      toast(err.message, 'error')
    }
  }

  if (!status) return <EmptyState>Loading…</EmptyState>

  if (hasSession) {
    return session ? (
      <Running session={session} onStop={stop} onAgain={stop} />
    ) : (
      <EmptyState>Starting…</EmptyState>
    )
  }
  return <Setup status={status} onStarted={refreshAll} />
}
