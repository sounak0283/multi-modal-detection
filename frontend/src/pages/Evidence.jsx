import { useCallback, useEffect, useMemo, useState } from 'react'
import { Badge, Button, Card, CardHead, Dot, EmptyState, Select, SeverityChip } from '../components/ui'
import { copyToClipboard, maskStoragePath } from '../lib/secure'
import { api, eventClipUrl, eventSnapshotUrl } from '../api'
import { useToast } from '../components/Toast'
import AlertAI from '../components/ai/AlertAI'

/* Evidence viewer: pick an alert, watch its video, read every detail about it.
 *
 * Playback goes through /api/events/{id}/clip, which (for S3 evidence) redirects to a fresh
 * short-lived signed link - so this page never holds a link that can expire underneath it,
 * and only logged-in users can watch. The permanent reference shown in the details is the
 * masked s3 path (never the real address); the signed link is only ever copied to the
 * clipboard via "Copy secure link", not displayed. */

const KIND_LABEL = {
  boundary: 'Boundary',
  fire: 'Fire',
  smoke: 'Smoke',
  crowd: 'Crowd',
  ppe: 'PPE',
  access: 'Unauthorized access',
}

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
          <Badge>{visible.length}</Badge>
        </CardHead>
        <div className="flex flex-wrap items-center gap-2 border-b border-ink-700 px-4 py-3">
          <Select value={filters.cameraId} onChange={set('cameraId')} className="w-auto!">
            <option value="">All cameras</option>
            {(cameras ?? []).map((camera) => (
              <option key={camera.id} value={camera.id}>
                {camera.name || camera.id}
              </option>
            ))}
          </Select>
          <Select value={filters.kind} onChange={set('kind')} className="w-auto!">
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
              className="accent-brand-500"
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
                      ? 'border-brand-500/40 bg-brand-900'
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
                    <span className="mt-1 flex items-center gap-1.5 font-mono text-[10.5px] text-ink-400">
                      <Dot tone={CLIP_STATE[event.clip_status]?.tone ?? (event.clip_path ? 'ok' : null)} />
                      {formatWhen(event.ts)} · {cameraName(event.camera_id)}
                      <SeverityChip severity={event.severity} className="ml-auto" />
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
    if (await copyToClipboard(text)) toast(`${what} copied.`, 'ok')
    else toast('Could not copy to the clipboard in this browser.', 'error')
  }

  return (
    <div className="space-y-4">
      <Card className="p-0">
        <div className="bg-black">
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
            className="text-brand-400 underline"
          >
            Open it in a new tab
          </a>
          . Clips saved before the H.264 fix use a format browsers cannot play inline.
        </p>
      )}

      <AlertAI event={event} />

      <Card>
        <CardHead title="Alert details">
          <div className="flex items-center gap-2">
            {clip && <Badge tone={clip.tone}>{clip.label}</Badge>}
            <SeverityChip severity={event.severity} />
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
            <MaskedField label="Stored at" text={maskStoragePath(event.clip_uri)} />
          )}
          {event.clip_path && (
            <div className="flex flex-wrap items-center gap-2">
              <Button
                onClick={() =>
                  linkStillValid
                    ? copy(event.clip_link_url, 'Secure link')
                    : copy(`${window.location.origin}${eventClipUrl(event.id)}`, 'Link')
                }
              >
                Copy secure link
              </Button>
              <span className="text-[11.5px] text-ink-400">
                {linkStillValid
                  ? `Expires ${formatWhen(withUtc(event.clip_link_expires_at), true)}.`
                  : 'Opens only for signed-in users; a fresh signed link is issued each time it is played.'}
              </span>
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between gap-3 border-b border-ink-800 py-1.5">
      <dt className="font-mono text-[11px] uppercase tracking-[0.06em] text-ink-400">{label}</dt>
      <dd className="text-right text-ink-100">{value}</dd>
    </div>
  )
}

/* Shows a masked storage path. The real address / signed link is never rendered; the
 * only way to get a link out is the "Copy secure link" button, straight to the clipboard. */
function MaskedField({ label, text }) {
  return (
    <div>
      <div className="mb-1 text-[11.5px] text-ink-400">{label}</div>
      <p className="break-all rounded-lg border border-ink-700 bg-ink-950 px-3 py-2 font-mono text-[11px] text-ink-300">
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
