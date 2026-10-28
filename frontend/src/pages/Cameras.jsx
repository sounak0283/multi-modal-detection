import { useCallback, useEffect, useState } from 'react'
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardFoot,
  CardHead,
  Dot,
  EmptyState,
  Field,
  Input,
  SegmentedControl,
  Select,
  TelemetryChip,
  TextField,
  Toggle,
  cx,
} from '../components/ui'
import { api, snapshotUrl } from '../api'
import { useToast } from '../components/Toast'

const RESOLUTIONS = [
  [640, 360],
  [1280, 720],
  [1920, 1080],
  [2560, 1440],
  [3840, 2160],
]

// Every module is opt-in, boundary included - a camera with none selected only captures
// and streams. Boundary is the person detection and tracking the modules in
// REQUIRES_BOUNDARY run on (mirrors backend cameras/models.py).
// `note` is the standing state shown on the module card: dependencies come from
// REQUIRES_BOUNDARY below, "needs a model" / "not built" mirror the notes this page has
// always carried (there is no backend endpoint reporting model availability per module).
const MODULE_OPTIONS = [
  { value: 'boundary', label: 'Boundary / restricted area' },
  { value: 'crowd', label: 'Crowd formation', note: 'Requires Boundary', tone: 'neutral' },
  { value: 'ppe', label: 'PPE compliance', note: 'Requires Boundary · needs a trained model on the server', tone: 'warn' },
  { value: 'phone', label: 'Phone near ear', note: 'Not built yet', tone: 'warn', unavailable: true },
  { value: 'fire_smoke', label: 'Fire & smoke' },
  { value: 'identity', label: 'Identity / face recognition', note: 'Requires Boundary', tone: 'neutral' },
  { value: 'welding', label: 'Welding sparks (soft alert)' },
]

// Off by default: a newly added camera should not start capturing/recording until an
// admin has actually configured its source and reviewed it. Requiring an explicit
// opt-in avoids an unconfigured or misconfigured camera silently going live.
const REQUIRES_BOUNDARY = ['crowd', 'ppe', 'phone', 'identity']

// Keeps the selection consistent both ways: picking a module that needs person tracking
// turns boundary on, and turning boundary off turns those modules off.
function withModuleDependencies(previous, next) {
  const boundaryRemoved = previous.includes('boundary') && !next.includes('boundary')
  if (boundaryRemoved) return next.filter((m) => !REQUIRES_BOUNDARY.includes(m))
  if (next.some((m) => REQUIRES_BOUNDARY.includes(m)) && !next.includes('boundary')) {
    return ['boundary', ...next]
  }
  return next
}

const BLANK_FORM = {
  id: '',
  name: '',
  source_type: 'webcam',
  source: '0',
  substream: '',
  width: 1280,
  height: 720,
  fps: 30,
  decode_fps: 12,
  autostart: true,
  enabled: false,
  enabled_modules: [],
}

// Client-side hygiene only - the backend accepts any non-empty id, but this id is used
// verbatim in URLs (/api/cameras/{id}/stream) and stored on every alert row, so spaces
// or mixed case make for confusing links and inconsistent-looking records.
const VALID_ID = /^[a-z0-9][a-z0-9_-]*$/

function formFromCamera(camera) {
  return {
    id: camera.id,
    name: camera.name,
    source_type: camera.source_type,
    source: String(camera.source ?? '0'),
    substream: camera.substream || '',
    width: camera.width,
    height: camera.height,
    fps: camera.fps,
    decode_fps: camera.decode_fps,
    autostart: camera.autostart,
    enabled: camera.enabled,
    enabled_modules: camera.enabled_modules,
  }
}

export default function Cameras({ devTools, onChanged, readOnly = false }) {
  const [cameras, setCameras] = useState([])
  const [selectedId, setSelectedId] = useState(null) // null | '_new' | an existing id
  const [form, setForm] = useState(null)
  const [rtsp, setRtsp] = useState('')
  const [stored, setStored] = useState({ set: false, redacted: '' })
  const [testing, setTesting] = useState(false)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [result, setResult] = useState(null)
  const [simulating, setSimulating] = useState(null)
  const [filter, setFilter] = useState('')
  const toast = useToast()

  const idValid = form ? VALID_ID.test(form.id.trim()) : false
  const nameValid = form ? form.name.trim().length > 0 : false
  const rtspOk = form && form.source_type === 'rtsp' ? stored.set || rtsp.trim().length > 0 : true
  const canSave = idValid && nameValid && rtspOk

  const load = useCallback(async () => {
    try {
      const body = await api.listCameras()
      setCameras(body.cameras ?? [])
    } catch (err) {
      toast(err.message, 'error')
    }
  }, [toast])

  useEffect(() => {
    load()
  }, [load])

  const selectExisting = (camera) => {
    setSelectedId(camera.id)
    setForm(formFromCamera(camera))
    setStored({ set: camera.rtsp_url_set, redacted: camera.rtsp_url_redacted })
    setRtsp('')
    setResult(null)
  }

  const startNew = () => {
    setSelectedId('_new')
    setForm({ ...BLANK_FORM })
    setStored({ set: false, redacted: '' })
    setRtsp('')
    setResult(null)
  }

  if (!form && cameras.length > 0 && selectedId === null) {
    // First load: default to editing the first camera rather than an empty panel.
    selectExisting(cameras[0])
  }

  const set = (patch) => setForm((f) => ({ ...f, ...patch }))

  // An empty RTSP field means "keep what is stored" - the browser is never sent the
  // real URL, so it cannot echo it back on save.
  const payload = () => {
    const base = { ...form, enabled_modules: form.enabled_modules }
    return rtsp.trim() ? { ...base, rtsp_url: rtsp.trim() } : base
  }

  const test = async () => {
    setTesting(true)
    setResult(null)
    try {
      setResult(await api.testCamera(payload()))
    } catch (err) {
      setResult({ ok: false, error: err.message })
    } finally {
      setTesting(false)
    }
  }

  const save = async () => {
    setSaving(true)
    try {
      const saved =
        selectedId === '_new'
          ? await api.createCamera(payload())
          : await api.updateCamera(selectedId, payload())
      toast(`Camera saved — ${saved.resolved}`, 'ok')
      setRtsp('')
      await load()
      selectExisting(saved)
      onChanged?.()
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setSaving(false)
    }
  }

  const remove = async () => {
    if (selectedId === '_new' || !selectedId) return
    if (
      !window.confirm(
        `Remove camera "${form.name || form.id}"? This also deletes its configured ` +
          'boundaries/zones, and cannot be undone.',
      )
    )
      return
    setDeleting(true)
    try {
      await api.deleteCamera(selectedId)
      toast('Camera removed.', 'ok')
      setSelectedId(null)
      setForm(null)
      await load()
      onChanged?.()
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setDeleting(false)
    }
  }

  const simulate = async (kind) => {
    setSimulating(kind)
    try {
      await api.simulateAlert(kind, selectedId)
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setSimulating(null)
    }
  }

  return (
    <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
      <Card>
        <CardHead title="Cameras" aside={<Badge>{cameras.length} configured</Badge>} />
        {!readOnly && (
          <div className="border-b border-ink-700 p-3">
            <Button variant="primary" className="w-full" onClick={startNew}>
              + Add camera
            </Button>
          </div>
        )}
        {cameras.length > 4 && (
          <div className="border-b border-ink-700 p-3">
            <Input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter cameras by name or ID"
            />
          </div>
        )}
        {cameras.length === 0 ? (
          <EmptyState>No cameras yet.</EmptyState>
        ) : (
          <ul className="space-y-1.5 p-2">
            {cameras
              .filter((camera) =>
                `${camera.name} ${camera.id}`.toLowerCase().includes(filter.trim().toLowerCase()),
              )
              .map((camera) => {
                const state = camera.running
                  ? { tone: 'ok', label: 'Running' }
                  : camera.enabled
                    ? { tone: 'warn', label: 'Enabled, not started' }
                    : { tone: 'neutral', label: 'Disabled' }
                return (
                  <li key={camera.id}>
                    <button
                      type="button"
                      onClick={() => selectExisting(camera)}
                      className={cx(
                        'flex w-full items-center gap-3 rounded-lg border p-2 text-left transition-colors',
                        selectedId === camera.id
                          ? 'border-brand-500/40 bg-brand-900'
                          : 'border-transparent hover:bg-ink-800',
                      )}
                    >
                      <CameraThumb camera={camera} />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[13px] font-medium">
                          {camera.name || camera.id}
                        </span>
                        <span className="mt-1 flex items-center gap-1.5">
                          <Badge tone={state.tone}>{state.label}</Badge>
                        </span>
                        <span className="mt-1 block truncate font-mono text-[10.5px] text-ink-400">
                          {camera.id} · {camera.width}×{camera.height} · {camera.fps} fps
                        </span>
                      </span>
                    </button>
                  </li>
                )
              })}
          </ul>
        )}
      </Card>

      {!form ? (
        <Card>
          <EmptyState>Add a camera to get started.</EmptyState>
        </Card>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2.5 px-1">
            <h2 className="text-[20px] font-semibold tracking-tight text-ink-100">
              {selectedId === '_new' ? 'New camera' : form.name || form.id}
            </h2>
            {selectedId !== '_new' && (
              <>
                <Badge tone={form.enabled ? 'ok' : 'neutral'}>
                  {form.enabled ? 'Enabled' : 'Disabled'}
                </Badge>
                <TelemetryChip label="ID" className="ml-auto">
                  {selectedId}
                </TelemetryChip>
              </>
            )}
          </div>

          {selectedId !== '_new' && <CameraPreview key={selectedId} camera={cameras.find((c) => c.id === selectedId)} />}

          {devTools && !readOnly && selectedId && selectedId !== '_new' && (
            <Card className="border-alarm-900">
              <CardHead
                title="Dev tools"
                aside={<span className="text-[11px] text-ink-400">PERIMETER_DEV_TOOLS=true</span>}
              />
              <CardBody className="flex flex-wrap items-center gap-3">
                <p className="mr-auto text-[12.5px] leading-relaxed text-ink-300">
                  Inject a fake fire/smoke detection on this camera so the on-screen alert
                  popup can be checked end to end — independent of whether a real
                  fire/smoke model is loaded for this camera.
                </p>
                <Button variant="danger" onClick={() => simulate('fire')} disabled={simulating !== null}>
                  {simulating === 'fire' ? 'Sending…' : 'Simulate fire alert'}
                </Button>
                <Button variant="danger" onClick={() => simulate('smoke')} disabled={simulating !== null}>
                  {simulating === 'smoke' ? 'Sending…' : 'Simulate smoke alert'}
                </Button>
              </CardBody>
            </Card>
          )}

          <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2">
            <Card>
              <CardHead title="Source" />
              <CardBody className="space-y-3.5">
                <Field
                  label="Camera ID"
                  hint={
                    form.id.trim() && !idValid
                      ? 'Lowercase letters, numbers, "_" and "-" only, starting with a letter or number.'
                      : 'Technical identifier used in the API and stored on every alert. Cannot be changed after creation.'
                  }
                >
                  <TextField
                    value={form.id}
                    onChange={(e) => set({ id: e.target.value })}
                    placeholder="cam_02"
                    locked={selectedId !== '_new'}
                    className={form.id.trim() && !idValid ? 'border-alarm-400' : undefined}
                  />
                </Field>
                <Field label="Display name" hint="Shown throughout the dashboard - live view, alerts, boundary editor.">
                  <Input
                    value={form.name}
                    onChange={(e) => set({ name: e.target.value })}
                    placeholder="Loading bay"
                  />
                </Field>

                <Field label="Source type">
                  <SegmentedControl
                    className="w-full [&>button]:flex-1"
                    value={form.source_type}
                    onChange={(source_type) => set({ source_type })}
                    options={[
                      { value: 'webcam', label: 'Webcam / file' },
                      { value: 'rtsp', label: 'CCTV / RTSP' },
                    ]}
                  />
                </Field>

                {form.source_type === 'rtsp' ? (
                  <Field
                    label="RTSP URL"
                    hint={
                      stored.set
                        ? `Stored: ${stored.redacted}. Leave blank to keep it, or type a new URL to replace it.`
                        : 'No URL stored yet.'
                    }
                  >
                    <Input
                      type="password"
                      autoComplete="off"
                      value={rtsp}
                      onChange={(e) => setRtsp(e.target.value)}
                      placeholder="rtsp://user:pass@192.168.1.64:554/stream1"
                    />
                  </Field>
                ) : (
                  <Field
                    label="Device index or video file"
                    hint="A number is a device index. Anything else is treated as a file path."
                  >
                    <Input value={form.source} onChange={(e) => set({ source: e.target.value })} />
                  </Field>
                )}

                <Toggle checked={form.enabled} label="Enabled" onChange={(enabled) => set({ enabled })} />
              </CardBody>
            </Card>

            <Card>
              <CardHead title="Capture" />
              <CardBody className="space-y-3.5">
                <Field label="Resolution">
                  <Select
                    value={`${form.width}x${form.height}`}
                    onChange={(e) => {
                      const [width, height] = e.target.value.split('x').map(Number)
                      set({ width, height })
                    }}
                  >
                    {RESOLUTIONS.map(([w, h]) => (
                      <option key={`${w}x${h}`} value={`${w}x${h}`}>
                        {w} × {h}
                      </option>
                    ))}
                    {!RESOLUTIONS.some(([w, h]) => w === form.width && h === form.height) && (
                      <option value={`${form.width}x${form.height}`}>
                        {form.width} × {form.height}
                      </option>
                    )}
                  </Select>
                </Field>

                <div className="grid grid-cols-2 gap-3">
                  <Field label="Camera fps">
                    <Input
                      type="number" min="1" max="240"
                      value={form.fps}
                      onChange={(e) => set({ fps: Number(e.target.value) })}
                    />
                  </Field>
                  <Field label="Decode fps">
                    <Input
                      type="number" min="1" max="120" step="0.5"
                      value={form.decode_fps}
                      onChange={(e) => set({ decode_fps: Number(e.target.value) })}
                    />
                  </Field>
                </div>

                <Toggle
                  checked={form.autostart}
                  label="Start this camera automatically"
                  onChange={(autostart) => set({ autostart })}
                />
              </CardBody>
            </Card>
          </div>

          <Card>
            <CardHead
              title="Detection modules"
              aside={
                <span className="text-[11px] text-ink-400">
                  {form.enabled_modules.length === 0
                    ? 'None selected - stream only'
                    : 'Only selected modules run'}
                </span>
              }
            />
            <CardBody>
              <div className="grid grid-cols-1 gap-2 md:grid-cols-2 2xl:grid-cols-3">
                {MODULE_OPTIONS.map((option) => {
                  const on = form.enabled_modules.includes(option.value)
                  const toggle = () =>
                    set({
                      enabled_modules: withModuleDependencies(
                        form.enabled_modules,
                        on
                          ? form.enabled_modules.filter((m) => m !== option.value)
                          : [...form.enabled_modules, option.value],
                      ),
                    })
                  return (
                    <button
                      key={option.value}
                      type="button"
                      aria-pressed={on}
                      onClick={toggle}
                      className={cx(
                        'flex items-start gap-3 rounded-lg border p-3 text-left transition-colors',
                        on
                          ? 'border-brand-500 bg-brand-900'
                          : 'border-ink-700 bg-ink-950 hover:border-ink-hover',
                      )}
                    >
                      <span
                        className={cx(
                          'mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-sm border text-[10px] leading-none',
                          on ? 'border-brand-500 bg-brand-500 text-white' : 'border-ink-500',
                        )}
                      >
                        {on && '✓'}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-[13px] font-medium text-ink-100">
                          {option.label}
                        </span>
                        {option.note && (
                          <span
                            className={cx(
                              'mt-1 block font-mono text-[10.5px] uppercase tracking-[0.04em]',
                              option.tone === 'warn' ? 'text-warn-400' : 'text-ink-400',
                            )}
                          >
                            {option.note}
                          </span>
                        )}
                      </span>
                    </button>
                  )
                })}
              </div>
              <p className="mt-3 text-[11.5px] leading-snug text-ink-400">
                Nothing runs unless you select it. Boundary / restricted area detects and
                tracks people for zone entry/exit alerts; crowd formation and identity/face
                recognition build on it and turn it on automatically. Fire &amp; smoke runs
                on its own. PPE compliance needs a trained model placed on the server
                before it does anything (see backend/models/ppe/README.md). Phone-near-ear
                detection is not built yet.
              </p>
            </CardBody>
            <CardFoot className="flex flex-wrap gap-2">
              <Button onClick={test} disabled={testing} className="flex-1">
                {testing ? 'Testing…' : 'Test connection'}
              </Button>
              {!readOnly && (
                <Button variant="primary" onClick={save} disabled={saving || !canSave} className="flex-1">
                  {saving
                    ? 'Saving…'
                    : selectedId === '_new'
                      ? 'Create camera'
                      : 'Save & reconnect'}
                </Button>
              )}
              {!readOnly && selectedId !== '_new' && (
                <Button variant="danger" onClick={remove} disabled={deleting}>
                  {deleting ? 'Removing…' : 'Remove'}
                </Button>
              )}
            </CardFoot>

            {!readOnly && !canSave && (
              <p className="mx-4 mb-4 text-[11.5px] text-warn-400">
                {!nameValid && 'A display name is required. '}
                {!idValid && form.id.trim() && 'Camera ID has invalid characters. '}
                {!idValid && !form.id.trim() && 'A camera ID is required. '}
                {!rtspOk && 'An RTSP URL is required for a CCTV / RTSP source.'}
              </p>
            )}

            {result && (
              <div
                className={`mx-4 mb-4 rounded-lg border-l-[3px] bg-ink-800 px-3.5 py-3 text-[12.5px] leading-relaxed ${
                  result.ok ? 'border-l-ok-400' : 'border-l-alarm-400'
                }`}
              >
                {result.ok ? (
                  <>
                    Connected in {result.elapsed_ms} ms.
                    <br />
                    Actual frame:{' '}
                    <code className="font-mono text-ink-100">
                      {result.actual_width}×{result.actual_height}
                    </code>
                    {result.reported_fps ? ` · reports ${result.reported_fps} fps` : ''}
                    {(result.warnings ?? []).map((warning) => (
                      <span key={warning} className="mt-1.5 block text-warn-400">
                        ⚠ {warning}
                      </span>
                    ))}
                  </>
                ) : (
                  result.error
                )}
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}

/* Still frame from the camera's own snapshot endpoint (the one Boundaries uses as its
 * backdrop). Taken once per mount rather than streamed: a list of live MJPEG tiles would
 * hold one connection per camera open for no benefit on a settings page. */
function useSnapshot(camera) {
  const [bust] = useState(() => Date.now())
  const [failed, setFailed] = useState(false)
  const src = camera?.running && !failed ? snapshotUrl(camera.id, bust) : null
  return { src, onError: () => setFailed(true) }
}

function CameraThumb({ camera }) {
  const { src, onError } = useSnapshot(camera)
  return (
    <span className="relative grid h-14 w-24 shrink-0 place-items-center overflow-hidden rounded-md border border-ink-700 bg-black">
      {src ? (
        <img src={src} onError={onError} alt="" className="h-full w-full object-cover" />
      ) : (
        <span className="font-mono text-[9.5px] uppercase tracking-[0.06em] text-ink-500">
          No signal
        </span>
      )}
    </span>
  )
}

function CameraPreview({ camera }) {
  const { src, onError } = useSnapshot(camera)
  if (!camera) return null
  return (
    <Card className="relative">
      <div className="grid aspect-[16/6] max-h-[300px] w-full place-items-center bg-black">
        {src ? (
          <img src={src} onError={onError} alt="Latest frame" className="h-full w-full object-contain" />
        ) : (
          <span className="font-mono text-[11px] uppercase tracking-[0.08em] text-ink-500">
            No signal
          </span>
        )}
      </div>
      <div className="absolute left-3 top-3 flex items-center gap-2">
        <span className="flex h-5 items-center gap-1.5 rounded-sm border border-ink-700 bg-black/70 px-1.5 font-mono text-[11px] uppercase text-white">
          <Dot tone={camera.running ? 'ok' : camera.enabled ? 'warn' : null} live={camera.running} />
          {camera.running ? 'Running' : camera.enabled ? 'Enabled, not started' : 'Disabled'}
        </span>
        <span className="flex h-5 items-center rounded-sm border border-ink-700 bg-black/70 px-1.5 font-mono text-[11px] text-white">
          {camera.width}×{camera.height} · {camera.fps} fps
        </span>
      </div>
    </Card>
  )
}
