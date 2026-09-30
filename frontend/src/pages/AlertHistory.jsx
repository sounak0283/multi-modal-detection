import { useCallback, useEffect, useState } from 'react'
import { Badge, Button, Card, CardFoot, CardHead, EmptyState, Select, Stat } from '../components/ui'
import { api, eventClipUrl, eventSnapshotUrl } from '../api'
import { emitsEvents } from '../lib/zones'

const KIND_LABEL = {
  boundary: 'Boundary',
  crowd: 'Crowd',
  fire: 'Fire',
  smoke: 'Smoke',
  ppe: 'PPE',
  access: 'Restricted access',
  welding: 'Welding sparks',
  health: 'Camera health',
}

const PERIODS = [
  ['today', 'Today'],
  ['week', 'This week'],
  ['month', 'This month'],
  ['6months', 'Last 6 months'],
  ['', 'All time'],
]
const PERIOD_LABEL = Object.fromEntries(PERIODS)

const DEFAULT_FILTERS = { kind: '', zoneId: '', cameraId: '', limit: 200, period: '' }

/** Start of a period in the viewer's own timezone, as an ISO timestamp (or '' = all time).
 * Recomputed on every refresh, so "Today" rolls over at the viewer's midnight. */
function periodStart(period, now = new Date()) {
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()) // local midnight
  if (period === 'today') return start.toISOString()
  if (period === 'week') {
    start.setDate(start.getDate() - ((start.getDay() + 6) % 7)) // back to Monday
    return start.toISOString()
  }
  if (period === 'month') return new Date(now.getFullYear(), now.getMonth(), 1).toISOString()
  if (period === '6months') {
    start.setMonth(start.getMonth() - 6)
    return start.toISOString()
  }
  return ''
}

export default function AlertHistory({ cameras, zones }) {
  const [filters, setFilters] = useState(DEFAULT_FILTERS)
  const [events, setEvents] = useState([])
  const [window_, setWindow] = useState(null) // counts per kind for period + camera + boundary
  const [error, setError] = useState(null)
  const [viewing, setViewing] = useState(null) // null | an event with evidence to show

  const load = useCallback(async () => {
    // One `since` for both requests, so the cards and the list count the same window.
    const since = periodStart(filters.period)
    try {
      const [page, totals] = await Promise.all([
        api.events({ ...filters, since }),
        api.eventsSummary(filters.cameraId, { zoneId: filters.zoneId, since }),
      ])
      setEvents(page.events)
      setWindow(totals.available ? totals.window || {} : null)
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

  const set = (key) => (value) => setFilters((f) => ({ ...f, [key]: value }))
  const active = Object.keys(DEFAULT_FILTERS).some(
    (key) => key !== 'limit' && filters[key] !== DEFAULT_FILTERS[key],
  )
  const stats = deriveStats(window_, filters, events.length)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {stats.map(({ label, value, tone }) => (
          <Stat key={label} label={label} value={value} tone={tone} />
        ))}
      </div>

      <Card>
        <div className="space-y-3 border-b border-ink-800 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <PeriodPills value={filters.period} onChange={set('period')} />
            <div className="flex items-center gap-2">
              {active && (
                <Button variant="ghost" onClick={() => setFilters(DEFAULT_FILTERS)}>
                  Clear filters
                </Button>
              )}
              <Button onClick={load}>Refresh</Button>
            </div>
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <FilterSelect label="Type" value={filters.kind} onChange={set('kind')}>
              <option value="">All types</option>
              {Object.entries(KIND_LABEL).map(([kind, label]) => (
                <option key={kind} value={kind}>
                  {label}
                </option>
              ))}
            </FilterSelect>
            <FilterSelect label="Boundary" value={filters.zoneId} onChange={set('zoneId')}>
              <option value="">All boundaries</option>
              {zones.filter(emitsEvents).map((zone) => (
                <option key={zone.id} value={zone.id}>
                  {zone.name || zone.id}
                </option>
              ))}
            </FilterSelect>
            <FilterSelect label="Camera" value={filters.cameraId} onChange={set('cameraId')}>
              <option value="">All cameras</option>
              {(cameras ?? []).map((camera) => (
                <option key={camera.id} value={camera.id}>
                  {camera.name || camera.id}
                </option>
              ))}
            </FilterSelect>
            <FilterSelect label="Show" value={filters.limit} onChange={set('limit')}>
              <option value="50">Newest 50</option>
              <option value="200">Newest 200</option>
              <option value="500">Newest 500</option>
            </FilterSelect>
          </div>
        </div>

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

        {events.length === 0 && (
          <EmptyState>
            {active ? 'No alerts match these filters.' : 'No alerts recorded yet.'}
          </EmptyState>
        )}

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
              <EmptyState>
                {event.clip_status === 'pending'
                  ? 'The video is still being saved - reopen this alert in a minute.'
                  : event.clip_status === 'failed'
                    ? 'The video could not be saved after several attempts.'
                    : 'No evidence was captured for this alert.'}
              </EmptyState>
            )}
            {/* Permanent storage reference for admin audit; playback links are minted
                fresh per view (they expire), so only the s3:// address is shown. */}
            {event.clip_uri && (
              <p className="break-all text-[11.5px] text-ink-400">
                Stored at <span className="font-mono text-ink-300">{event.clip_uri}</span>
              </p>
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
  if (event.kind === 'health' || event.kind === 'welding') return 'info'
  return event.subtype === 'entry' ? 'ok' : 'warn'
}

/** The cards count exactly what the list is filtered to: same period, camera and boundary
 * (the server applies one shared `since`). "Listed" makes the Show cap visible - the list
 * holds the newest N of the matching alerts, never silently fewer. */
function deriveStats(counts, filters, listed) {
  const period = PERIOD_LABEL[filters.period]
  if (!counts) {
    return [
      { label: `Alerts · ${period}`, value: '—' },
      { label: 'Boundary', value: '—' },
      { label: 'Fire / smoke', value: '—' },
      { label: 'Listed below', value: listed },
    ]
  }
  const total = Object.values(counts).reduce((a, b) => a + b, 0)
  const fireSmoke = (counts.fire || 0) + (counts.smoke || 0)
  const matching = filters.kind ? counts[filters.kind] || 0 : total
  return [
    { label: `Alerts · ${period}`, value: total },
    { label: `Boundary · ${period}`, value: counts.boundary || 0 },
    {
      label: `Fire / smoke · ${period}`,
      value: fireSmoke,
      tone: fireSmoke > 0 ? 'alarm' : undefined,
    },
    { label: 'Listed below', value: matching > listed ? `${listed} of ${matching}` : listed },
  ]
}

function PeriodPills({ value, onChange }) {
  return (
    <div
      role="radiogroup"
      aria-label="Period"
      className="inline-flex flex-wrap overflow-hidden rounded-lg border border-ink-700"
    >
      {PERIODS.map(([period, label]) => (
        <button
          key={period || 'all'}
          type="button"
          role="radio"
          aria-checked={value === period}
          onClick={() => onChange(period)}
          className={`px-3.5 py-1.5 text-[12.5px] transition-colors not-first:border-l not-first:border-ink-700 ${
            value === period
              ? 'bg-brand-500 font-semibold text-on-brand'
              : 'bg-ink-800 text-ink-200 hover:bg-ink-700'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

function FilterSelect({ label, value, onChange, children }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[10.5px] font-semibold uppercase tracking-[0.08em] text-ink-400">
        {label}
      </span>
      <Select value={value} onChange={(event) => onChange(event.target.value)}>
        {children}
      </Select>
    </label>
  )
}
