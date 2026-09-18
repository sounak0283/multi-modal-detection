import { useCallback, useEffect, useState } from 'react'
import { Badge, Button, Card, CardFoot, CardHead, EmptyState, Select, Stat } from '../components/ui'
import { api, eventClipUrl, eventSnapshotUrl } from '../api'
import { emitsEvents } from '../lib/zones'

const KIND_LABEL = { boundary: 'Boundary', fire: 'Fire', smoke: 'Smoke', health: 'Camera' }

export default function AlertHistory({ cameras, zones }) {
  const [filters, setFilters] = useState({ kind: '', zoneId: '', cameraId: '', limit: 200 })
  const [events, setEvents] = useState([])
  const [summary, setSummary] = useState({ available: false, counts: {}, today: {} })
  const [error, setError] = useState(null)
  const [viewing, setViewing] = useState(null) // null | an event with evidence to show

  const load = useCallback(async () => {
    try {
      const [page, totals] = await Promise.all([
        api.events(filters),
        api.eventsSummary(filters.cameraId),
      ])
      setEvents(page.events)
      setSummary(totals)
      setError(null)
    } catch (err) {
      setError(err.message)
    }
  }, [filters])

  useEffect(() => {
    load()
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [load])

  const stats = deriveStats(summary)
  const set = (key) => (event) => setFilters((f) => ({ ...f, [key]: event.target.value }))

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {stats.map(({ label, value, tone }) => (
          <Stat key={label} label={label} value={value} tone={tone} />
        ))}
      </div>

      <Card>
        <CardHead title="Alerts">
          <div className="flex flex-wrap gap-2">
            <Select value={filters.kind} onChange={set('kind')} className="w-auto py-1.5">
              <option value="">All types</option>
              <option value="boundary">Boundary</option>
              <option value="fire">Fire</option>
              <option value="smoke">Smoke</option>
              <option value="health">Camera health</option>
            </Select>
            <Select value={filters.zoneId} onChange={set('zoneId')} className="w-auto py-1.5">
              <option value="">All boundaries</option>
              {zones.filter(emitsEvents).map((zone) => (
                <option key={zone.id} value={zone.id}>
                  {zone.name || zone.id}
                </option>
              ))}
            </Select>
            <Select value={filters.cameraId} onChange={set('cameraId')} className="w-auto py-1.5">
              <option value="">All cameras</option>
              {(cameras ?? []).map((camera) => (
                <option key={camera.id} value={camera.id}>
                  {camera.name || camera.id}
                </option>
              ))}
            </Select>
            <Select value={filters.limit} onChange={set('limit')} className="w-auto py-1.5">
              <option value="50">Last 50</option>
              <option value="200">Last 200</option>
              <option value="500">Last 500</option>
            </Select>
            <Button onClick={load}>Refresh</Button>
          </div>
        </CardHead>

        {/* Wide content scrolls inside its own box; the page never scrolls sideways. */}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="bg-ink-800">
                {['Time', 'Alert', 'Boundary', 'Event', 'Track', 'Severity', 'Evidence'].map((head) => (
                  <th
                    key={head}
                    className="whitespace-nowrap px-4 py-2.5 text-left text-[10.5px] font-semibold uppercase tracking-[0.08em] text-ink-400"
                  >
                    {head}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {events.map((event) => (
                <tr key={event.id ?? event.ts} className="border-t border-ink-800 hover:bg-ink-800">
                  <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-ink-300">
                    {formatWhen(event.ts)}
                  </td>
                  <td className="min-w-[210px] px-4 py-2.5 font-medium">{event.message}</td>
                  <td className="whitespace-nowrap px-4 py-2.5 text-ink-200">
                    {event.zone_name || event.zone_id || '—'}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5">
                    <Badge tone={toneFor(event)}>
                      {event.subtype || KIND_LABEL[event.kind] || event.kind}
                    </Badge>
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-ink-300">
                    {event.track_id != null ? `#${event.track_id}` : '—'}
                  </td>
                  <td
                    className={`whitespace-nowrap px-4 py-2.5 ${
                      { high: 'font-semibold text-alarm-400', medium: 'text-warn-400' }[
                        event.severity
                      ] || 'text-ink-400'
                    }`}
                  >
                    {event.severity || '—'}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5">
                    {event.snapshot_path ? (
                      <button
                        type="button"
                        onClick={() => setViewing(event)}
                        className="block h-10 w-14 overflow-hidden rounded-md border border-ink-700 transition-colors hover:border-brand-500"
                      >
                        <img
                          src={eventSnapshotUrl(event.id)}
                          alt="Alert snapshot"
                          className="h-full w-full object-cover"
                        />
                      </button>
                    ) : (
                      <span className="text-ink-500">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {events.length === 0 && <EmptyState>No alerts recorded yet.</EmptyState>}

        <CardFoot>
          <p className="text-[11.5px] text-ink-400">
            {error ? `Could not load history: ${error}` : 'Stored in MongoDB.'}
          </p>
        </CardFoot>
      </Card>

      {viewing && <EvidenceLightbox event={viewing} onClose={() => setViewing(null)} />}
    </div>
  )
}

/* Full snapshot + clip playback for one alert (Expansion Plan Phase C). A plain
 * `Card`-based overlay, same look `FireAlertOverlay.jsx` already established for a
 * modal over the whole dashboard. */
function EvidenceLightbox({ event, onClose }) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
    >
      {/* Card itself does not spread onClick, so the stop-propagation guard lives on a
          plain wrapper - clicking anywhere inside the card must not bubble up to the
          overlay's own onClick and close the lightbox. */}
      <div className="w-full max-w-xl" onClick={(event_) => event_.stopPropagation()}>
        <Card>
          <CardHead
            title={event.message || 'Alert evidence'}
            aside={<span className="text-[11.5px] text-ink-400">{formatWhen(event.ts)}</span>}
          >
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
          </CardHead>
          <div className="space-y-3 p-4">
            {event.snapshot_path && (
              <img
                src={eventSnapshotUrl(event.id)}
                alt="Alert snapshot"
                className="w-full rounded-lg border border-ink-800"
              />
            )}
            {event.clip_path && (
              // eslint-disable-next-line jsx-a11y/media-has-caption
              <video
                src={eventClipUrl(event.id)}
                controls
                className="w-full rounded-lg border border-ink-800"
              />
            )}
            {!event.snapshot_path && !event.clip_path && (
              <EmptyState>No evidence was captured for this alert.</EmptyState>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}

function formatWhen(iso) {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  // Local time: an alert log is read against wall-clock memory ("what happened just
  // before 2am"), so the viewer's own zone is the useful one.
  return date.toLocaleString(undefined, {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

function toneFor(event) {
  if (event.kind === 'fire' || event.kind === 'smoke') return 'alarm'
  if (event.kind === 'health') return 'info'
  return event.subtype === 'entry' ? 'ok' : 'warn'
}

function deriveStats(summary) {
  const sum = (counts) => Object.values(counts).reduce((a, b) => a + b, 0)
  const today = summary.today || {}
  const fireSmokeToday = (today.fire || 0) + (today.smoke || 0)

  return [
    { label: 'Today', value: sum(today) },
    { label: 'Boundary today', value: today.boundary || 0 },
    { label: 'Fire / smoke today', value: fireSmokeToday, tone: fireSmokeToday > 0 ? 'alarm' : undefined },
    { label: 'All time', value: sum(summary.counts || {}) },
  ]
}
