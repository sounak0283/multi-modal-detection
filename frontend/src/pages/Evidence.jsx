import { useCallback, useEffect, useMemo, useState } from 'react'
import { Badge, Button, Card, CardHead, Dot, EmptyState, Select } from '../components/ui'
import { api, eventClipUrl, eventSnapshotUrl } from '../api'
import { useToast } from '../components/Toast'

/* Evidence viewer: pick an alert, watch its video, read every detail about it.
 *
 * Playback goes through /api/events/{id}/clip, which (for S3 evidence) redirects to a fresh
 * short-lived signed link - so this page never holds a link that can expire underneath it,
 * and only logged-in users can watch. The permanent reference shown in the details is the
 * s3:// address; the emailed link is shown only while it is still valid. */

const KIND_LABEL = {
  boundary: 'Boundary',
  fire: 'Fire',
  smoke: 'Smoke',
  crowd: 'Crowd',
  ppe: 'PPE',
  access: 'Unauthorized access',
}

const SEVERITY_TONE = { critical: 'alarm', high: 'warn', medium: 'info', low: 'neutral' }

const CLIP_STATE = {
  saved: { tone: 'ok', label: 'Video saved' },
  pending: { tone: 'warn', label: 'Saving video…' },
  failed: { tone: 'alarm', label: 'Video failed' },
  skipped: { tone: 'neutral', label: 'Not recorded' },
}

const hasEvidence = (e) => Boolean(e.clip_path || e.snapshot_path || e.clip_status)

export default function Evidence({ cameras }) {
  const [filters, setFilters] = useState({ cameraId: '', kind: '', onlyVideo: true })
  const [events, setEvents] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    try {
      const page = await api.events({
        cameraId: filters.cameraId,
        kind: filters.kind,
        limit: 200,
      })
      setEvents(page.events.filter(hasEvidence))
      setError(null)
    } catch (err) {
      setError(err.message)
    }
  }, [filters.cameraId, filters.kind])

  useEffect(() => {
    load()
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [load])

  const visible = useMemo(
    () => (filters.onlyVideo ? events.filter((e) => e.clip_path) : events),
    [events, filters.onlyVideo],
  )

  // Keep the current selection across refreshes; fall back to the newest row.
  const selected = visible.find((e) => e.id === selectedId) ?? visible[0] ?? null

  const set = (key) => (event) => setFilters((f) => ({ ...f, [key]: event.target.value }))
  const cameraName = (id) => cameras?.find((c) => c.id === id)?.name || id

  return (
    <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[380px_minmax(0,1fr)]">
      <Card>
        <CardHead title="Alerts with evidence">
          <span className="rounded-full bg-ink-700 px-2 py-0.5 text-[11px] text-ink-200">
            {visible.length}
          </span>
        </CardHead>
        <div className="flex flex-wrap items-center gap-2 border-b border-ink-800 px-4 py-3">
          <Select value={filters.cameraId} onChange={set('cameraId')} className="w-auto py-1.5">
            <option value="">All cameras</option>
            {(cameras ?? []).map((camera) => (
              <option key={camera.id} value={camera.id}>
                {camera.name || camera.id}
              </option>
            ))}
          </Select>
          <Select value={filters.kind} onChange={set('kind')} className="w-auto py-1.5">
            <option value="">All types</option>
            {Object.entries(KIND_LABEL).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </Select>
          <label className="flex items-center gap-1.5 text-[12px] text-ink-300">
            <input
              type="checkbox"
              checked={filters.onlyVideo}
              onChange={(e) => setFilters((f) => ({ ...f, onlyVideo: e.target.checked }))}
            />
            Video only
          </label>
        </div>

        {visible.length === 0 ? (
          <EmptyState>
            {error ? `Could not load alerts: ${error}` : 'No alerts with saved evidence yet.'}
          </EmptyState>
        ) : (
          <ul className="max-h-[calc(100vh-260px)] overflow-y-auto p-1.5">
            {visible.map((event) => (
              <li key={event.id}>
                <button
                  type="button"
                  onClick={() => setSelectedId(event.id)}
                  className={`flex w-full items-center gap-3 rounded-lg border px-2 py-2 text-left transition-colors ${
                    selected?.id === event.id
                      ? 'border-brand-600 bg-brand-900'
                      : 'border-transparent hover:bg-ink-800'
                  }`}
                >
                  <span className="h-11 w-16 shrink-0 overflow-hidden rounded-md border border-ink-700 bg-ink-800">
                    {event.snapshot_path && (
                      <img
                        src={eventSnapshotUrl(event.id)}
                        alt=""
                        loading="lazy"
                        className="h-full w-full object-cover"
                      />
                    )}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[12.5px] text-ink-100">{event.message}</span>
                    <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-ink-400">
                      <Dot tone={CLIP_STATE[event.clip_status]?.tone ?? (event.clip_path ? 'ok' : null)} />
                      {formatWhen(event.ts)} · {cameraName(event.camera_id)}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>

      {selected ? (
        <Detail event={selected} cameraName={cameraName(selected.camera_id)} />
      ) : (
        <Card>
          <EmptyState>Select an alert to see its video and details.</EmptyState>
        </Card>
      )}
    </div>
  )
}

function Detail({ event, cameraName }) {
  const toast = useToast()
  const clip = CLIP_STATE[event.clip_status]
  const linkStillValid =
    event.clip_link_url &&
    event.clip_link_expires_at &&
    new Date(withUtc(event.clip_link_expires_at)) > new Date()

  const copy = async (text, what) => {
    try {
      await navigator.clipboard.writeText(text)
      toast(`${what} copied.`, 'ok')
    } catch {
      toast('Could not copy - select the text and copy it manually.', 'error')
    }
  }

  return (
    <div className="space-y-4">
      <Card className="p-0">
        <div className="bg-[#05070a]">
          {event.clip_path ? (
            // eslint-disable-next-line jsx-a11y/media-has-caption
            <video
              key={event.id}
              src={eventClipUrl(event.id)}
              poster={event.snapshot_path ? eventSnapshotUrl(event.id) : undefined}
              controls
              preload="metadata"
              className="mx-auto block max-h-[60vh] w-full"
            />
          ) : event.snapshot_path ? (
            <img
              src={eventSnapshotUrl(event.id)}
              alt="Alert snapshot"
              className="mx-auto block max-h-[60vh] w-full object-contain"
            />
          ) : (
            <EmptyState>
              {event.clip_status === 'pending'
                ? 'The video is still being saved - this page refreshes on its own.'
                : event.clip_status === 'failed'
                  ? 'The video could not be saved after several attempts.'
                  : event.clip_status === 'skipped'
                    ? 'No video by design: a registered person entering, or an exit. Change this in config/app.yaml under storage.clip_triggers.'
                    : 'No video was captured for this alert.'}
            </EmptyState>
          )}
        </div>
      </Card>

      {event.clip_path && (
        <p className="px-1 text-[11.5px] text-ink-400">
          Video not playing?{' '}
          <a
            href={eventClipUrl(event.id)}
            target="_blank"
            rel="noreferrer"
            className="text-brand-500 underline"
          >
            Open it in a new tab
          </a>
          . Clips saved before the H.264 fix use a format browsers cannot play inline.
        </p>
      )}

      <Card>
        <CardHead title="Alert details">
          <div className="flex items-center gap-2">
            {clip && <Badge tone={clip.tone}>{clip.label}</Badge>}
            <Badge tone={SEVERITY_TONE[event.severity] ?? 'neutral'}>{event.severity}</Badge>
          </div>
        </CardHead>
        <div className="space-y-3 p-4">
          <p className="text-[14px] font-medium text-ink-100">{event.message}</p>
          <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-[12.5px] sm:grid-cols-2">
            <Row label="Time" value={formatWhen(event.ts, true)} />
            <Row label="Camera" value={cameraName} />
            <Row label="Type" value={KIND_LABEL[event.kind] ?? event.kind} />
            <Row label="Event" value={event.subtype || '—'} />
            <Row label="Boundary" value={event.zone_name || event.zone_id || '—'} />
            <Row label="Track" value={event.track_id != null ? `#${event.track_id}` : '—'} />
            <Row label="Person" value={personLabel(event)} />
            <Row
              label="Match confidence"
              value={event.identity_confidence != null ? `${Math.round(event.identity_confidence * 100)}%` : '—'}
            />
            <Row label="Email" value={event.delivered ? 'Sent' : 'Not recorded as sent'} />
            <Row label="Storage" value={event.evidence_backend || '—'} />
          </dl>

          {event.clip_uri && (
            <Copyable
              label="Stored at (permanent)"
              text={event.clip_uri}
              onCopy={() => copy(event.clip_uri, 'Address')}
            />
          )}
          {linkStillValid ? (
            <Copyable
              label={`Emailed link (valid until ${formatWhen(withUtc(event.clip_link_expires_at), true)})`}
              text={event.clip_link_url}
              onCopy={() => copy(event.clip_link_url, 'Link')}
            />
          ) : (
            event.clip_link_url && (
              <p className="text-[11.5px] text-ink-400">
                The emailed link has expired. Use the player above - it signs a fresh link each time.
              </p>
            )
          )}
        </div>
      </Card>
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between gap-3 border-b border-ink-800 py-1.5">
      <dt className="text-ink-400">{label}</dt>
      <dd className="text-right text-ink-100">{value}</dd>
    </div>
  )
}

function Copyable({ label, text, onCopy }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11.5px] text-ink-400">
        <span>{label}</span>
        <Button variant="ghost" onClick={onCopy} className="px-2 py-0.5 text-[11.5px]">
          Copy
        </Button>
      </div>
      <p className="break-all rounded-lg border border-ink-800 bg-ink-800 px-3 py-2 font-mono text-[11px] text-ink-300">
        {text}
      </p>
    </div>
  )
}

function personLabel(event) {
  if (event.identity_status === 'known') return event.identity_name || 'Known person'
  if (event.identity_status === 'unknown_face') return 'Unrecognised'
  if (event.identity_status === 'no_face') return 'Face not visible'
  return '—'
}

// Mongo datetimes arrive without a zone suffix; they are UTC.
function withUtc(iso) {
  return /Z|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`
}

function formatWhen(iso, withSeconds = false) {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    ...(withSeconds ? { second: '2-digit' } : {}),
    hour12: false,
  })
}
