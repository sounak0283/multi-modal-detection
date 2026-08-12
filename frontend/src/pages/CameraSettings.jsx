import { useEffect, useState } from 'react'
import {
  Button,
  Card,
  CardBody,
  CardFoot,
  CardHead,
  Field,
  Input,
  Select,
  Toggle,
} from '../components/ui'
import { api } from '../api'
import { useToast } from '../components/Toast'

const RESOLUTIONS = [
  [640, 360],
  [1280, 720],
  [1920, 1080],
  [2560, 1440],
  [3840, 2160],
]

export default function CameraSettings({ onSaved }) {
  const [form, setForm] = useState(null)
  const [rtsp, setRtsp] = useState('')
  const [stored, setStored] = useState({ set: false, redacted: '' })
  const [actual, setActual] = useState(null)
  const [testing, setTesting] = useState(false)
  const [saving, setSaving] = useState(false)
  const [result, setResult] = useState(null)
  const toast = useToast()

  useEffect(() => {
    api
      .getCamera()
      .then((camera) => {
        setForm({
          use_cctv: camera.use_cctv,
          source: camera.source ?? '0',
          id: camera.id ?? 'cam_01',
          width: camera.width,
          height: camera.height,
          fps: camera.fps,
          decode_fps: camera.decode_fps,
          autostart: camera.autostart,
        })
        setStored({ set: camera.rtsp_url_set, redacted: camera.rtsp_url_redacted })
        setActual(camera.actual_size)
      })
      .catch((err) => toast(err.message, 'error'))
  }, [toast])

  if (!form) return <p className="text-[13px] text-ink-400">Loading camera settings…</p>

  const set = (patch) => setForm((f) => ({ ...f, ...patch }))

  // An empty RTSP field means "keep what is stored". The browser is never sent the real
  // URL - it cannot be read back, only replaced - so it cannot echo it on save.
  const payload = () => (rtsp.trim() ? { ...form, rtsp_url: rtsp.trim() } : form)

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
      const saved = await api.saveCamera(payload())
      toast(`Camera saved — reconnecting to ${saved.resolved}`, 'ok')
      setRtsp('')
      onSaved?.()
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setSaving(false)
    }
  }

  const mismatched = actual?.[0] && (actual[0] !== form.width || actual[1] !== form.height)
  const detectionHz = form.decode_fps ? (form.decode_fps / 2).toFixed(1) : null

  return (
    <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2">
      <Card>
        <CardHead title="Source" />
        <CardBody className="space-y-3.5">
          <Field label="Camera type">
            <div className="flex overflow-hidden rounded-lg border border-ink-700">
              {[
                [false, 'Laptop webcam'],
                [true, 'CCTV / RTSP'],
              ].map(([value, label]) => (
                <button
                  key={label}
                  type="button"
                  data-cctv={String(value)}
                  onClick={() => set({ use_cctv: value })}
                  className={`flex-1 px-3 py-2 text-[12.5px] transition-colors not-first:border-l not-first:border-ink-700 ${
                    form.use_cctv === value
                      ? 'bg-brand-500 font-semibold text-[#04122a]'
                      : 'bg-ink-800 text-ink-200 hover:bg-ink-700'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </Field>

          {form.use_cctv ? (
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

          <Field label="Camera name" hint="Appears in alerts and in the boundary configuration.">
            <Input value={form.id} onChange={(e) => set({ id: e.target.value })} />
          </Field>
        </CardBody>
      </Card>

      <Card>
        <CardHead title="Capture" />
        <CardBody className="space-y-3.5">
          <Field
            label="Resolution"
            hint={
              mismatched
                ? `Camera is actually streaming ${actual[0]}×${actual[1]}. Pixel-based thresholds are tuned for the configured size.`
                : undefined
            }
          >
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
                type="number"
                min="1"
                max="240"
                value={form.fps}
                onChange={(e) => set({ fps: Number(e.target.value) })}
              />
            </Field>
            <Field label="Decode fps">
              <Input
                type="number"
                min="1"
                max="120"
                step="0.5"
                value={form.decode_fps}
                onChange={(e) => set({ decode_fps: Number(e.target.value) })}
              />
            </Field>
          </div>

          {detectionHz && (
            <p className="text-[11.5px] leading-snug text-ink-400">
              Detection runs at ~{detectionHz} Hz. Boundary hysteresis is counted in these
              frames, not camera frames.
            </p>
          )}

          <Toggle
            checked={form.autostart}
            label="Start the camera automatically"
            onChange={(autostart) => set({ autostart })}
          />
        </CardBody>

        <CardFoot className="flex gap-2">
          <Button onClick={test} disabled={testing} className="flex-1">
            {testing ? 'Testing…' : 'Test connection'}
          </Button>
          <Button variant="primary" onClick={save} disabled={saving} className="flex-1">
            {saving ? 'Saving…' : 'Save & reconnect'}
          </Button>
        </CardFoot>

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
  )
}
